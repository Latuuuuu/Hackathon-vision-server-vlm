const $ = id => document.getElementById(id);
let choiceKey = '', uiEpoch = 0, serverGeneration = null;
let shown = null;
const lastKeys = {raw:'', tracked:''};
const urls = {raw:null, tracked:null};
const n = (v,d=1) => v == null ? '—' : Number(v).toFixed(d);

function clearShown() {
  shown = null;
  lastKeys.tracked = '';
  $('tracked').classList.add('hidden');
  $('tracked').removeAttribute('src');
}
async function post(path, data={}) {
  uiEpoch++;
  clearShown();
  try {
    const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j = await r.json();
    if (!r.ok) throw Error(j.error);
    $('error').textContent = '';
  } catch(e) { $('error').textContent = e.message; }
}
$('detect').onclick = () => post('/api/detect',{target:$('target').value});
$('stop').onclick = () => post('/api/stop');

// One outstanding request per view. Every response is the server's latest
// completed frame; no MJPEG/browser stream backlog is carried forward.
async function frameLoop(kind) {
  const epoch = uiEpoch, started = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(()=>controller.abort(),5000);
  let newUrl = null;
  try {
    const r = await fetch('/frame/'+kind+'?after='+encodeURIComponent(lastKeys[kind]),
                          {cache:'no-store',signal:controller.signal});
    if (r.status === 204) return;
    if (!r.ok) throw Error('HTTP '+r.status);
    const key = r.headers.get('X-Frame-Key');
    const gen = Number(r.headers.get('X-Generation'));
    const age = Number(r.headers.get('X-Frame-Age-Ms'));
    const blob = await r.blob();
    newUrl = URL.createObjectURL(blob);
    const decoded = new Image();
    decoded.src = newUrl;
    await decoded.decode();
    if (epoch !== uiEpoch || (serverGeneration !== null && gen !== serverGeneration)) return;
    $(kind).src = newUrl;
    if (urls[kind]) URL.revokeObjectURL(urls[kind]);
    urls[kind] = newUrl;
    newUrl = null;
    lastKeys[kind] = key;
    if (kind === 'tracked') {
      shown = {key,gen,at:performance.now(),age:age+performance.now()-started};
      $('tracked').classList.remove('hidden');
    }
  } catch(e) {
    // Status polling reports connectivity; retain the last displayed frame and
    // continue increasing its age instead of claiming it is live.
  } finally {
    clearTimeout(timer);
    if (newUrl) URL.revokeObjectURL(newUrl);
    setTimeout(()=>frameLoop(kind),kind==='raw'?30:45);
  }
}

setInterval(()=>{
  const age = shown ? shown.age+performance.now()-shown.at : null;
  $('age').textContent = age===null ? '等待最新追蹤結果' : '目前顯示影格距今約 '+n(age/1000)+' 秒';
  $('age').className = age>500 ? 'stale' : '';
},100);

async function poll() {
  try {
    const s = await (await fetch('/api/status',{cache:'no-store'})).json();
    serverGeneration = s.generation;
    if (shown && (shown.gen!==s.generation || s.output_captured===null)) clearShown();
    $('state').textContent = s.phase+' / '+s.camera;
    $('error').textContent = s.error||s.camera_error||'';
    const stage = s.stage_ms||{};
    const values = [['相機 FPS',n(s.camera_fps)],['SAM 完成 FPS',n(s.tracking_fps)],
      ['SAM 單步 ms',n(s.sam_step_ms)],['Locate ms',n(s.locate_ms)],
      ['影像編碼 ms',n(stage.image_encoder_ms)],['記憶注意力 ms',n(stage.memory_attention_ms)],
      ['遮罩解碼 ms',n(stage.mask_decoder_ms)],['記憶編碼 ms',n(stage.memory_encoder_ms)],
      ['單步 P95 ms',n(s.step_p95_ms)],['完成時影格年齡 ms',n(s.completion_age_ms)],
      ['遮罩像素',n(s.mask_pixels,0)],['略過相機影格',n(s.skipped_camera_frames,0)]];
    $('metrics').replaceChildren(...values.map(([label,value])=>{
      const e=document.createElement('div'),a=document.createElement('small'),b=document.createElement('span');
      a.textContent=label;b.className='value';b.textContent=value;e.append(a,b);return e;
    }));
    s.browser_display = shown ? {frame_key:shown.key,approx_age_ms:shown.age+performance.now()-shown.at} : null;
    $('details').textContent = JSON.stringify(s,null,2);
    const show=s.phase==='choose';$('candidates').hidden=!show;
    if(show){
      const key=s.generation+':'+JSON.stringify(s.boxes);
      if(key!==choiceKey){
        choiceKey=key;$('candidate').src='/candidate.jpg?v='+s.generation;
        $('choices').replaceChildren(...s.boxes.map((b,i)=>{
          const e=document.createElement('button');e.textContent='追蹤 '+(i+1);
          e.onclick=()=>post('/api/select',{generation:s.generation,index:i});return e;
        }));
      }
    }
  } catch(e) { $('state').textContent='伺服器未連線：'+e.message; }
  finally { setTimeout(poll,500); }
}
poll();frameLoop('raw');frameLoop('tracked');
