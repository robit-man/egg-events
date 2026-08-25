from __future__ import annotations

import asyncio
import mimetypes
import re
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

from egg_companion.config import EggConfig
from egg_companion.runtime import CompanionRuntime
from egg_companion.services.audit import AuditCheck, audit_hardware
from egg_companion.services.dashboard_ui import PAGE as APPLICATION_PAGE


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Egg · Companion</title><style>
:root{color-scheme:dark;--bg:#070a0c;--panel:#0d1318;--panel-hi:#101b22;--line:#263740;--ink:#e4eff1;--muted:#82939a;--mint:#8ee8d3;--blue:#a7caff;--warn:#ffad72;--bad:#ff7f8f}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#17455070,transparent 35%),var(--bg);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Menlo,monospace}.shell{max-width:1680px;margin:auto;padding:18px}.top,.telemetry{display:flex;align-items:center;justify-content:space-between;gap:14px}.brand{font-size:19px;font-weight:800;letter-spacing:.18em}.sub,.detail{color:var(--muted);font-size:11px}.status{border:1px solid var(--mint);color:var(--mint);padding:6px 9px;text-transform:uppercase;letter-spacing:.08em}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:10px;margin-top:14px}.panel{position:relative;overflow:hidden;padding:12px;min-height:90px;background:linear-gradient(145deg,#101a21e8,#090d11e8);border:1px solid var(--line)}.panel:before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,#ffffff05 1px,transparent 1px),linear-gradient(#ffffff04 1px,transparent 1px);background-size:20px 20px;pointer-events:none}.panel>*{position:relative}.optics{grid-column:span 8}.side-stack{grid-column:span 4;display:grid;gap:10px;align-content:start}.full{grid-column:1/-1}.title{color:var(--blue);font-size:11px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px}.cameras{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.camera{overflow:hidden;border:1px solid var(--line);background:#030506}.camera-head{display:flex;justify-content:space-between;gap:8px;padding:7px 8px}.camera-stage{position:relative;overflow:hidden;background:#020304;isolation:isolate}.camera-raw{display:block;width:100%;height:100%;object-fit:fill}.camera-overlay{position:absolute;inset:0;pointer-events:none}.mask-layer{position:absolute;inset:0;width:100%;height:100%;overflow:visible}.mask{fill:#8ee8d355;stroke:#c5fff4;stroke-width:2;vector-effect:non-scaling-stroke}.mask-label{fill:#eafffb;font-size:18px;font-weight:700;paint-order:stroke;stroke:#071116;stroke-width:5;stroke-linejoin:round}.camera-meta{min-height:34px;align-items:center}.chips{display:flex;flex-wrap:wrap;gap:5px}.chip{border:1px solid #2e5962;color:var(--mint);padding:3px 5px;font-size:10px}.inference-stamp{color:var(--muted);font-size:10px}.wave{display:block;width:100%;height:110px;background:#04080b;border:1px solid var(--line)}.voice,.memory-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.voice label,.memory-controls label{display:grid;gap:5px;color:var(--muted);font-size:10px}.voice input,.voice select,.memory-controls input{width:100%;min-width:0;padding:8px;background:#080e12;border:1px solid var(--line);color:var(--ink);font:inherit}.button{padding:8px;background:#123a40;border:1px solid #34717a;color:var(--mint);font:inherit;cursor:pointer}.button.danger{border-color:#713747;background:#38151d;color:#ffb8c1}.transcript{white-space:pre-wrap;line-height:1.55}.identities{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px}.identity{padding:5px;border:1px solid var(--line);color:var(--muted);font-size:10px}.identity img{display:block;width:100%;aspect-ratio:1;object-fit:cover;margin-bottom:4px;background:#05080b}.seen{max-height:180px;overflow:auto}.memory-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}.memory-list{max-height:240px;overflow:auto}.memory-row{padding:7px;border-bottom:1px solid #18242b}.check{display:grid;grid-template-columns:8px 1fr;gap:8px;padding:7px 0;border-bottom:1px solid #18242b}.dot{width:7px;height:7px;margin-top:4px;background:var(--bad)}.pass .dot{background:var(--mint)}.warn .dot{background:var(--warn)}.empty{padding:18px;color:var(--muted)}@media(max-width:1050px){.optics,.side-stack{grid-column:1/-1}}@media(max-width:700px){.cameras,.voice,.memory-controls,.memory-grid{grid-template-columns:1fr}.shell{padding:10px}}
</style></head><body><main class="shell"><header class="top"><div><div class="brand">EGG / COMPANION</div><div class="sub">live sensory field · associative memory · local cognition</div></div><div id="status" class="status">CHECKING</div></header><section class="grid">
<article class="panel optics"><div class="title">Optical Array · Raw Streams + Instance Masks</div><div id="cameras" class="cameras"><div class="empty">Awaiting camera streams</div></div></article>
<div class="side-stack"><article class="panel"><div class="title">Audio Input</div><canvas id="wave" class="wave"></canvas><div class="telemetry"><span class="sub">ReSpeaker live waveform</span><span id="asr-state" class="sub">idle</span></div><div id="asr-metrics" class="chips" style="margin-top:8px"></div></article>
<article class="panel"><div class="title">Conversation</div><div id="conversation" class="transcript sub">No local speech captured yet.</div></article>
<article class="panel"><div class="title">Voice Controls</div><form id="voice" class="voice"><label>ASR ENGINE<select name="asr_model"></select></label><label>TTS ENGINE<select name="voice_model"></select></label><label>TTS VOICE<select name="voice_name"></select></label><label>SEGMENT SEC<input name="segment_seconds" type="number" min="1" max="15" step=".5"></label><label>RMS GATE<input name="rms_threshold" type="number" min=".001" max="1" step=".001"></label><button class="button">APPLY LIVE</button></form><div id="voice-result" class="sub" style="margin-top:10px"></div></article>
<article class="panel"><div class="title">People</div><div id="identities" class="identities"><span class="sub">Awaiting validated face crops.</span></div></article>
<article class="panel"><div class="title">Identity Features</div><div id="identity-features" class="chips seen"><span class="sub">Awaiting identity evidence.</span></div></article>
<article class="panel"><div class="title">Segmented Objects</div><div id="object-learning-state" class="chips" style="margin-bottom:8px"></div><div id="review-queue-state" class="chips" style="margin-bottom:8px"></div><div id="objects" class="identities"><span class="sub">Hold an object up and identify it naturally.</span></div></article>
<article class="panel"><div class="title">Seen Library</div><div id="seen" class="chips seen"><span class="sub">No scene categories yet.</span></div></article></div>
<article class="panel full"><div class="title">Cognition · Evidence + Action Ledger</div><div id="cognition" class="memory-grid"><div class="empty">Awaiting cognition state.</div></div></article>
<article class="panel full"><div class="title">GPU / VRAM · jetson-stats (ground truth, not self-reported)</div><div id="gpu-stats" class="chips" style="margin-bottom:8px"></div><div id="gpu-processes" class="detail transcript">Awaiting jetson-stats.</div></article>
<article class="panel full"><div class="title">Associative Memory · Governance</div><div class="memory-grid"><div><div id="memory-stats" class="chips"></div><div id="memory-entities" class="memory-list"></div><div id="memory-jobs" class="sub"></div><pre id="memory-inspector" class="detail transcript">Select an entity to inspect evidence and claims.</pre></div><form id="memory-controls" class="memory-controls"><label>ENTITY ID<input name="entity_id" placeholder="person-001 or object-001"></label><label>ALIAS<input name="alias" placeholder="user-provided alias"></label><button class="button" name="action" value="alias">ADD ALIAS</button><button class="button danger" name="action" value="delete" type="button" id="delete-memory">DELETE ENTITY</button><label>CLAIM ID<input name="claim_id" placeholder="claim UUID"></label><label>REPLACEMENT<input name="replacement" placeholder="corrected value"></label><button class="button" name="action" value="correct">CORRECT CLAIM</button><button class="button" type="button" id="export-memory">EXPORT JSON</button><div id="memory-result" class="sub"></div></form></div></article>
<article class="panel full"><div class="title">Readiness</div><div id="checks"></div></article></section></main><script>
const $=s=>document.querySelector(s),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let catalog=null,lastPeopleSignature='',lastObjectSignature='';const cameraViews=new Map();
function drawWave(values){const canvas=$('#wave'),ctx=canvas.getContext('2d'),width=canvas.width=canvas.clientWidth*devicePixelRatio,height=canvas.height=canvas.clientHeight*devicePixelRatio;ctx.clearRect(0,0,width,height);ctx.strokeStyle='#8ee8d3';ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();(values.length?values:[0]).forEach((value,index)=>{const x=index*width/Math.max(1,values.length-1),y=height/2-value*height*.44;index?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
function select(name,options,value){const node=$(`#voice [name=${name}]`),signature=JSON.stringify(options);if(node.dataset.options!==signature){node.innerHTML=options.map(option=>`<option value="${esc(option.id)}">${esc(option.label)}</option>`).join('');node.dataset.options=signature}if(document.activeElement!==node)node.value=value??options[0]?.id??''}
function voiceChoices(state){if(!catalog)return;const voice=state.telemetry?.voice||{},tts=(catalog.tts?.models||[]).filter(model=>model.enabled!==false).map(model=>({id:model.id,label:`${model.label||model.id} · ${model.backend||'tts'}`})),asr=(catalog.asr?.models||[]).map(model=>({id:model.id,label:model.id+(model.isActive?' · active':'')}));select('voice_model',tts,voice.tts_model);select('asr_model',asr,voice.asr_model);const selected=tts.find(model=>model.id===$('#voice [name=voice_model]').value)?.id,voices=selected==='supertonic'?(catalog.supertonic?.options?.voices||[]).map(id=>({id,label:id})):[];select('voice_name',voices,voice.tts_voice||catalog.supertonic?.settings?.voiceName);$('#voice [name=voice_name]').disabled=!voices.length}
function cameraView(camera){let view=cameraViews.get(camera.id);if(view)return view;if(!cameraViews.size)$('#cameras').replaceChildren();const card=document.createElement('div');card.className='camera';const head=document.createElement('div');head.className='camera-head';const title=document.createElement('b'),rate=document.createElement('span');rate.className='sub';head.append(title,rate);const stage=document.createElement('div');stage.className='camera-stage';const raw=document.createElement('img');raw.className='camera-raw';raw.alt=`${camera.id} raw live stream`;raw.decoding='async';const overlay=document.createElement('div');overlay.className='camera-overlay';stage.append(raw,overlay);const meta=document.createElement('div');meta.className='camera-head camera-meta';card.append(head,stage,meta);$('#cameras').append(card);view={card,title,rate,stage,raw,overlay,meta};cameraViews.set(camera.id,view);return view}
function maskSvg(detections,width,height){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('class','mask-layer');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('preserveAspectRatio','none');for(const detection of detections){const polygon=detection.mask_polygon;if(!Array.isArray(polygon)||polygon.length<3)continue;const points=polygon.map(point=>[Number(point[0]),Number(point[1])]).filter(point=>point.every(Number.isFinite));if(points.length<3)continue;const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('class','mask');path.setAttribute('d',`M ${points.map(point=>point.join(' ')).join(' L ')} Z`);svg.append(path);const anchor=points.reduce((current,point)=>point[1]<current[1]?point:current,points[0]);const text=document.createElementNS(svg.namespaceURI,'text');text.setAttribute('class','mask-label');text.setAttribute('x',String(anchor[0]));text.setAttribute('y',String(Math.max(18,anchor[1]-6)));const confidence=Math.round(Number(detection.identity_confidence??detection.confidence??0)*100);text.textContent=`${detection.identity||detection.label||'object'} ${confidence}%${detection.behavior?` · ${detection.behavior}`:''}`;svg.append(text)}return svg}
function renderCameras(cameras){const present=new Set(cameras.map(camera=>camera.id));for(const [id,view] of cameraViews)if(!present.has(id)){view.card.remove();cameraViews.delete(id)}if(!cameras.length){$('#cameras').innerHTML='<div class="empty">Awaiting discovered V4L2 streams</div>';return}for(const camera of cameras){const view=cameraView(camera),shape=Array.isArray(camera.frame_shape)?camera.frame_shape:[],height=Number(shape[0]),width=Number(shape[1]);view.title.textContent=camera.id;view.rate.textContent=`${camera.resolved_rotation===null?'CALIBRATING':camera.resolved_rotation+'°'} · ${camera.fps??'--'} RAW · ${camera.inference_fps??'--'} MASK FPS`;view.stage.style.aspectRatio=width>0&&height>0?`${width}/${height}`:'16/9';if(camera.raw_stream_url&&view.raw.dataset.stream!==camera.raw_stream_url){view.raw.src=camera.raw_stream_url;view.raw.dataset.stream=camera.raw_stream_url}view.overlay.replaceChildren(maskSvg(camera.detections||[],width||16,height||9));const labels=document.createElement('div');labels.className='chips';for(const label of camera.semantic_labels||[]){const chip=document.createElement('span');chip.className='chip';chip.textContent=label;labels.append(chip)}if(!labels.childElementCount){const none=document.createElement('span');none.className='sub';none.textContent='Inference pending';labels.append(none)}const stamp=document.createElement('span');stamp.className='inference-stamp';stamp.textContent=camera.detections_updated_at?`mask #${camera.detection_sequence||0} · ${camera.detections_updated_at.slice(11,19)}`:'mask pending';view.meta.replaceChildren(labels,stamp)}}
function renderPeople(identities){const signature=JSON.stringify(identities.slice(0,12).map(identity=>[identity.id,identity.label,identity.kind,identity.confidence,identity.samples,identity.sightings,identity.last_seen]));if(signature===lastPeopleSignature)return;lastPeopleSignature=signature;$('#identities').innerHTML=identities.slice(0,12).map(identity=>`<div class="identity"><img src="${esc(identity.thumbnail_url)}?t=${encodeURIComponent(identity.last_seen)}" alt="${esc(identity.label)} face crop"><b>${esc(identity.label||identity.id)}</b><br>${esc(identity.kind||'face')} · ${Math.round((identity.confidence||0)*100)}% · ${esc(identity.samples)} samples · seen ${esc(identity.sightings||0)}×</div>`).join('')||'<span class="sub">Awaiting validated face crops.</span>'}
function renderIdentityGroups(state){const identities=state.identities||[],objects=state.objects||[];$('#identity-features').innerHTML=identities.map(identity=>`<span class="chip">${esc(identity.label)} · ${esc(identity.kind||'appearance')} · ${Math.round((identity.confidence||0)*100)}%</span>`).join('')||'<span class="sub">Awaiting identity evidence.</span>';const signature=JSON.stringify(objects.map(object=>[object.id,object.label,object.label_confidence,object.confidence,object.samples,object.last_match_state,object.review_state,object.label_source,object.last_seen,(object.label_history||[]).length]));if(signature===lastObjectSignature)return;lastObjectSignature=signature;$('#objects').innerHTML=objects.map(object=>`<div class="identity"><img src="${esc(object.thumbnail_url)}?t=${encodeURIComponent(object.last_seen)}" alt="${esc(object.label)} segmented object"><b>${esc(object.label)}</b><br>${Math.round((object.label_confidence||object.confidence||0)*100)}% · ${esc(object.samples)} samples<br>${esc(object.last_match_state||object.review_state||'pending')} · ${esc(object.label_source||'legacy')}<br>${esc((object.label_history||[]).length)} prior label(s)${object.label_provenance?.ocr?.text?`<br>OCR: ${esc(object.label_provenance.ocr.text.slice(0,60))}`:''}</div>`).join('')||'<span class="sub">Awaiting sparse masked-object learning.</span>'}
function renderMemory(memory){const stats=memory?.stats||{};$('#memory-stats').innerHTML=Object.entries(stats).map(([key,value])=>`<span class="chip">${esc(key)} ${esc(value)}</span>`).join('')||'<span class="sub">Memory unavailable.</span>';$('#memory-entities').innerHTML=(memory?.entities||[]).map(entity=>`<button type="button" class="memory-row button" data-entity="${esc(entity.entity_id)}"><b>${esc(entity.display_name||entity.entity_id)}</b> · ${esc(entity.entity_type)}<div class="detail">${esc(entity.entity_id)} · ${esc(entity.state)} · ${esc(entity.updated_at)}</div></button>`).join('')||'<div class="empty">No graph entities yet.</div>';const jobs=memory?.jobs||[],conflicts=memory?.claim_conflicts||[],buffer=memory?.transient_buffer||{};$('#memory-jobs').textContent=`jobs ${jobs.length?jobs[0].state:'none'} · unresolved claim conflicts ${conflicts.length} · transient refs ${(buffer.frame_references||0)+(buffer.audio_references||0)}`}
function renderCognition(telemetry){const lifecycle=telemetry.memory?.lifecycle||{},active=lifecycle.active||[],boundary=lifecycle.last_boundary||{},attention=(telemetry.attention_decisions||[]).slice(-1)[0],interaction=(telemetry.interaction_decisions||[]).slice(-1)[0],hits=telemetry.retrieval_hits||[],errors=telemetry.runtime_errors||[],brain=telemetry.brain||{},sensing=brain.sensing||{},brainCognition=brain.cognition||{};$('#cognition').innerHTML=`<div class="chips" style="grid-column:1/-1;margin-bottom:8px"><span class="chip">SENSING · ${esc(sensing.target_count??0)} target(s) · ${esc(sensing.top_label||'none')}${sensing.top_priority!=null?' '+Math.round(sensing.top_priority*100)+'%':''}</span><span class="chip">COGNITION · capture ${brainCognition.capture_priority!=null?Math.round(brainCognition.capture_priority*100)+'%':'—'} · ${esc(brainCognition.allow_outward_speech?'SPEECH ALLOWED':'INTERNAL')}</span><span class="chip">MEMORY · active episodes ${active.length} · boundary ${esc(boundary.reason||'pending')}</span></div><div><div class="chips"><span class="chip">active episodes ${active.length}</span><span class="chip">boundary ${esc(boundary.reason||'pending')}</span><span class="chip">retrieval hits ${hits.length}</span></div><div class="detail">${esc(attention?.reason||'No attention decision')} · capture ${esc(attention?.capture_priority??'—')}</div></div><div><div class="detail">${esc(interaction?.allowed?'SPOKEN':'SUPPRESSED')} · ${esc(interaction?.reason||'No interaction decision')}</div><div class="detail">${esc(errors.length?`${errors.at(-1).component}: ${errors.at(-1).detail}`:'No runtime errors')}</div></div>`}
function renderGpu(gpu){if(!gpu||!gpu.updated_at){$('#gpu-stats').innerHTML='<span class="sub">Awaiting jetson-stats.</span>';return}$('#gpu-stats').innerHTML=`<span class="chip">RAM ${esc(gpu.ram_used_mb)}/${esc(gpu.ram_total_mb)} MB</span><span class="chip">GPU load ${gpu.gpu_load_percent!=null?esc(gpu.gpu_load_percent)+'%':'—'}</span>`;$('#gpu-processes').innerHTML=(gpu.processes||[]).map(process=>`<div>${esc(process.pid)} · ${esc(process.name)} · ${esc(process.gpu_memory_mb)} MB GPU · ${esc(process.memory_mb)} MB RSS · ${esc(process.cpu_percent)}% CPU</div>`).join('')||'<span class="sub">No GPU-resident processes reported.</span>'}
function render(state){$('#status').textContent=state.runtime.toUpperCase();const telemetry=state.telemetry||{},voice=telemetry.voice||{},vad=telemetry.vad||{},asr=telemetry.asr||{},learning=telemetry.object_learning||{};renderCameras(telemetry.cameras||[]);$('#asr-state').textContent=`${voice.asr_input||'ASR input'} · VAD ${vad.speech?'SPEECH':'SILENCE'} ${Math.round(Number(vad.speech_ratio||0)*100)}%/${Number(vad.speech_ms||0)}ms · RMS ${Number(telemetry.audio_rms||0).toFixed(4)} · ${telemetry.latest_transcript_at?`LAST ${telemetry.latest_transcript_at.slice(11,19)}`:'LISTENING'}`;$('#asr-metrics').innerHTML=`<span class="chip">accepted ${esc(asr.accepted||0)}</span><span class="chip">rejected ${esc(asr.rejected||0)}</span><span class="chip">errors ${esc(asr.errors||0)}</span>${asr.last_rejection?`<span class="chip">last reject ${esc(asr.last_rejection)}</span>`:''}`;$('#conversation').innerHTML=`<b>HEARD #${esc(telemetry.transcript_count||0)}</b> ${esc(telemetry.latest_transcript||'—')}\n\n<b>EGG</b> ${esc(telemetry.latest_reply||'—')}`;$('#object-learning-state').innerHTML=`<span class="chip">stable ${esc(learning.stable_candidates||0)}</span><span class="chip">CLIP ${esc(learning.clip_recalls||0)}/${esc(learning.clip_queries||0)}</span><span class="chip">VLM ${esc(learning.vlm_successes||0)}/${esc(learning.vlm_requests||0)}</span><span class="chip">OCR ${esc(learning.ocr_hits||0)}/${esc(learning.ocr_requests||0)}</span><span class="chip">rejected ${esc(learning.vlm_rejections||0)}</span><span class="chip">${esc(learning.last_stage||'idle')}</span>`;$('#review-queue-state').innerHTML=`<span class="chip">queue ${esc(learning.review_queue_depth||0)}</span><span class="chip">audited ok ${esc(learning.audit_consistent||0)}</span><span class="chip">flagged ${esc(learning.audit_flagged||0)}</span><span class="chip">failures ${esc(learning.vlm_errors||0)}</span>`;renderPeople(state.identities||[]);$('#seen').innerHTML=(telemetry.seen||[]).map(item=>`<span class="chip">${esc(item.label)} ×${esc(item.count)}</span>`).join('')||'<span class="sub">No scene categories observed.</span>';for(const [key,value] of Object.entries({segment_seconds:voice.asr_segment_seconds,rms_threshold:voice.asr_rms_threshold}))if(document.activeElement!==$(`#voice [name=${key}]`))$(`#voice [name=${key}]`).value=value??'';voiceChoices(state);renderIdentityGroups(state);renderCognition(telemetry);renderGpu(telemetry.gpu);renderMemory(state.memory);$('#checks').innerHTML=(state.checks||[]).map(check=>`<div class="check ${esc(check.status)}"><span class="dot"></span><div><b>${esc(check.name)}</b><div class="detail">${esc(check.detail)}</div></div></div>`).join('')}
async function refresh(){try{render(await fetch('/api/state',{cache:'no-store'}).then(response=>response.json()))}catch(_){$('#status').textContent='OFFLINE'}}async function loadCatalog(){catalog=await fetch('/api/voice/catalog').then(response=>response.json())}$('#voice [name=voice_model]').addEventListener('change',()=>voiceChoices({telemetry:{voice:{tts_model:$('#voice [name=voice_model]').value}}}));$('#voice').addEventListener('submit',async event=>{event.preventDefault();const response=await fetch('/api/voice/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(event.target)))});$('#voice-result').textContent=response.ok?'APPLIED':await response.text();if(response.ok){catalog=null;await loadCatalog();await refresh()}});$('#memory-entities').addEventListener('click',async event=>{const row=event.target.closest('[data-entity]');if(!row)return;const response=await fetch(`/api/memory/entities/${encodeURIComponent(row.dataset.entity)}`);$('#memory-inspector').textContent=response.ok?JSON.stringify(await response.json(),null,2):await response.text()});$('#memory-controls').addEventListener('submit',async event=>{event.preventDefault();const data=Object.fromEntries(new FormData(event.target)),response=await fetch(`/api/memory/entities/${encodeURIComponent(data.entity_id)}/aliases`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alias:data.alias})});$('#memory-result').textContent=response.ok?'ALIAS APPENDED':await response.text();await refresh()});$('#memory-controls [name=action][value=correct]').addEventListener('click',async event=>{event.preventDefault();const data=Object.fromEntries(new FormData($('#memory-controls'))),response=await fetch(`/api/memory/claims/${encodeURIComponent(data.claim_id)}/correct`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({replacement:data.replacement})});$('#memory-result').textContent=response.ok?'CLAIM REVISED':await response.text();await refresh()});$('#delete-memory').addEventListener('click',async()=>{const id=$('#memory-controls [name=entity_id]').value;if(!id||!confirm(`Delete ${id} from graph and profile libraries?`))return;const response=await fetch(`/api/memory/entities/${encodeURIComponent(id)}`,{method:'DELETE'});$('#memory-result').textContent=response.ok?'ENTITY DELETED':await response.text();await refresh()});$('#export-memory').addEventListener('click',()=>window.open('/api/memory/export','_blank'));function connectLiveWaveform(){const protocol=location.protocol==='https:'?'wss':'ws',socket=new WebSocket(`${protocol}://${location.host}/api/audio/stream`);socket.addEventListener('message',event=>{try{drawWave(JSON.parse(event.data).samples||[])}catch(_){}});socket.addEventListener('close',()=>setTimeout(connectLiveWaveform,500));socket.addEventListener('error',()=>socket.close())}loadCatalog().then(refresh).catch(refresh);setInterval(refresh,1000);connectLiveWaveform();
</script></body></html>"""


PAGE = APPLICATION_PAGE


class ReadinessMonitor:
    """Non-blocking recurring audit which replaces stale failures on recovery."""

    def __init__(
        self,
        config: EggConfig,
        *,
        probe: Callable[[EggConfig], Awaitable[list[AuditCheck]]] | None = None,
        healthy_interval_seconds: float = 300.0,
        degraded_interval_seconds: float = 30.0,
    ) -> None:
        self.config = config
        self.probe = probe or audit_hardware
        self.healthy_interval_seconds = healthy_interval_seconds
        self.degraded_interval_seconds = degraded_interval_seconds
        self.checks: list[AuditCheck] = []
        self._task: asyncio.Task[list[AuditCheck]] | None = None
        self._next_probe_at = 0.0
        self._updated_at: str | None = None

    async def _run_probe(self) -> list[AuditCheck]:
        try:
            return await self.probe(self.config)
        except Exception as error:
            return [
                AuditCheck(
                    "hardware-audit", "warn", f"diagnostic probe failed: {error}"
                )
            ]

    async def poll(self) -> list[AuditCheck]:
        now = asyncio.get_running_loop().time()
        if self._task is not None and self._task.done():
            self.checks = self._task.result()
            self._task = None
            self._updated_at = datetime.now(timezone.utc).isoformat()
            degraded = any(check.status != "pass" for check in self.checks)
            self._next_probe_at = now + (
                self.degraded_interval_seconds
                if degraded
                else self.healthy_interval_seconds
            )
        if self._task is None and now >= self._next_probe_at:
            self._task = asyncio.create_task(
                self._run_probe(), name="hardware-audit"
            )
        return list(self.checks)

    def snapshot(self) -> dict[str, object]:
        now = asyncio.get_running_loop().time()
        return {
            "probing": self._task is not None,
            "updated_at": self._updated_at,
            "next_probe_seconds": round(max(0.0, self._next_probe_at - now), 1),
        }


async def serve_dashboard(config: EggConfig, port: int) -> None:
    dashboard_executor = ThreadPoolExecutor(
        max_workers=6, thread_name_prefix="egg-dashboard"
    )

    async def dashboard_call(function, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(dashboard_executor, function, *args)

    async def measured_dashboard_call(name: str, function, *args):
        started = time.monotonic()
        value = await dashboard_call(function, *args)
        return name, value, round((time.monotonic() - started) * 1000, 1)

    readiness = ReadinessMonitor(config)
    state: dict[str, object] = {
        "runtime": "checking",
        "checks": [],
        "runtime_task": None,
        "companion": None,
    }

    async def refresh() -> None:
        state["checks"] = await readiness.poll()
        task = state["runtime_task"]
        if isinstance(task, asyncio.Task) and not task.done():
            checks = state["checks"]
            degraded = isinstance(checks, list) and any(
                isinstance(check, AuditCheck) and check.status != "pass" for check in checks
            )
            state["runtime"] = "live (degraded)" if degraded else "live"
            return
        if isinstance(task, asyncio.Task) and task.done():
            error = task.exception()
            state["checks"] = [
                *state["checks"],
                AuditCheck("runtime", "warn", f"stopped and restarting: {error}"),
            ]
            state["runtime_task"] = None
            state["companion"] = None
        state["runtime"] = "starting companion"
        try:
            runtime = CompanionRuntime(config)
        except Exception as error:
            state["runtime"] = "degraded: startup retry pending"
            state["checks"] = [
                *state["checks"],
                AuditCheck("runtime-initialization", "warn", f"retrying after: {error}"),
            ]
            return
        state["companion"] = runtime
        state["runtime_task"] = asyncio.create_task(runtime.run())

    def companion() -> CompanionRuntime:
        runtime = state["companion"]
        if not isinstance(runtime, CompanionRuntime):
            raise web.HTTPConflict(text="companion runtime is not active")
        return runtime

    def require_loopback(request: web.Request) -> None:
        if request.remote not in {"127.0.0.1", "::1", None}:
            raise web.HTTPForbidden(text="memory mutations are restricted to the local device")

    async def state_handler(_: web.Request) -> web.Response:
        await refresh()
        runtime = state["companion"]
        dashboard_timings: dict[str, float] = {}
        if isinstance(runtime, CompanionRuntime):
            measured = await asyncio.gather(
                measured_dashboard_call("telemetry", runtime.telemetry.snapshot, config),
                measured_dashboard_call("identities", runtime.identities.dashboard_snapshot),
                measured_dashboard_call("dreams", runtime.dreams_summary_snapshot),
                measured_dashboard_call("objects", runtime.objects.dashboard_snapshot),
                measured_dashboard_call("memory", runtime.memory_snapshot),
            )
            values = {name: value for name, value, _ in measured}
            dashboard_timings = {name: duration for name, _, duration in measured}
            telemetry = values["telemetry"]
            identities, identity_summary = values["identities"]
            dreams = values["dreams"]
            objects = values["objects"]
            memory = values["memory"]
        else:
            telemetry = {"cameras": []}
            identities, identity_summary, dreams, objects, memory = [], {}, {}, [], {}
        return web.json_response({
            "runtime": state["runtime"],
            "checks": [{"name": check.name, "status": check.status, "detail": check.detail} for check in state["checks"]],
            "readiness": readiness.snapshot(),
            "omnius": str(config.omnius.base_url),
            "telemetry": telemetry,
            "identities": identities,
            "identity_summary": identity_summary,
            "dreams": dreams,
            "objects": objects,
            "memory": memory,
            "dashboard_timings_ms": dashboard_timings,
        })

    async def index_handler(_: web.Request) -> web.Response:
        return web.Response(
            text=PAGE,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def config_handler(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "config": config.model_dump(mode="json"),
                "live_mutable": [
                    "transcription.segment_seconds",
                    "transcription.rms_threshold",
                    "transcription.asr_model",
                    "transcription.asr_language",
                    "transcription.vad_input_gain",
                    "audio.asr_target_rms",
                    "audio.asr_max_gain",
                    "omnius.voice_model",
                    "omnius.voice_name",
                ],
            },
            headers={"Cache-Control": "no-store"},
        )

    async def graph_handler(request: web.Request) -> web.Response:
        try:
            node_limit = max(50, min(2000, int(request.query.get("limit", "1500"))))
        except ValueError as error:
            raise web.HTTPBadRequest(text="limit must be an integer") from error
        try:
            payload = await dashboard_call(companion().knowledge_graph_snapshot, node_limit)
        except Exception as error:
            import traceback, logging
            logging.getLogger(__name__).error("graph_handler error: %s\n%s", error, traceback.format_exc())
            raise web.HTTPInternalServerError(text=f"graph error: {error}") from error
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def graph_node_handler(request: web.Request) -> web.Response:
        kind = request.query.get("kind", "")
        source_id = request.query.get("id", "")
        if kind not in {"entity", "evidence", "claim", "episode"} or not source_id:
            raise web.HTTPBadRequest(text="kind and id identify a graph node")
        detail = await dashboard_call(companion().graph_node_detail, kind, source_id)
        if detail is None:
            raise web.HTTPNotFound(text="graph node is not available")
        return web.json_response(detail, headers={"Cache-Control": "no-store"})

    async def evidence_media_handler(request: web.Request) -> web.Response:
        artifact = await dashboard_call(
            companion().evidence_media, request.match_info["evidence_id"]
        )
        if artifact is None:
            raise web.HTTPNotFound(text="evidence artifact is not retained")
        payload, suffix = artifact
        content_type = mimetypes.types_map.get(suffix, "application/octet-stream")
        return web.Response(
            body=payload,
            content_type=content_type,
            headers={"Cache-Control": "private, max-age=60"},
        )

    async def raw_frame_handler(request: web.Request) -> web.Response:
        frame = companion().telemetry.frame(request.match_info["camera_id"])
        if frame is None:
            raise web.HTTPNotFound(text="raw camera frame is not available yet")
        return web.Response(body=frame, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def camera_stream_handler(request: web.Request) -> web.StreamResponse:
        camera_id = request.match_info["camera_id"]
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=eggframe",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(request)
        sequence = -1
        try:
            while True:
                snapshot = companion().telemetry.frame_snapshot(camera_id)
                if snapshot is not None and snapshot[0] != sequence:
                    sequence, frame = snapshot
                    await response.write(
                        b"--eggframe\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        + frame
                        + b"\r\n"
                    )
                await asyncio.sleep(1 / 120)
        except (asyncio.CancelledError, ConnectionResetError, RuntimeError):
            pass
        return response

    async def audio_stream_handler(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        sequence = -1
        try:
            while not socket.closed:
                payload = companion().telemetry.waveform_snapshot()
                current = int(payload["sequence"])
                if current != sequence:
                    await socket.send_json(payload)
                    sequence = current
                await asyncio.sleep(1 / 120)
        finally:
            await socket.close()
        return socket

    async def identity_handler(request: web.Request) -> web.Response:
        face = companion().identities.thumbnail(request.match_info["profile_id"])
        if face is None:
            raise web.HTTPNotFound(text="identity crop is not available")
        return web.Response(body=face, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def identity_sample_handler(request: web.Request) -> web.Response:
        face = companion().identities.face_sample(
            request.match_info["profile_id"], request.match_info["sample_id"]
        )
        if face is None:
            raise web.HTTPNotFound(text="identity evidence crop is not available")
        return web.Response(
            body=face,
            content_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400"},
        )

    async def identity_timeline_handler(request: web.Request) -> web.Response:
        timeline = await dashboard_call(
            companion().identity_timeline, request.match_info["profile_id"]
        )
        if timeline is None:
            raise web.HTTPNotFound(text="identity timeline is not available")
        return web.json_response(timeline, headers={"Cache-Control": "no-store"})

    async def object_handler(request: web.Request) -> web.Response:
        object_crop = companion().objects.thumbnail(request.match_info["profile_id"])
        if object_crop is None:
            raise web.HTTPNotFound(text="segmented object crop is not available")
        return web.Response(body=object_crop, content_type="image/png", headers={"Cache-Control": "no-store"})

    async def voice_catalog_handler(request: web.Request) -> web.Response:
        return web.json_response(
            await companion()._omnius.voice_catalog(
                force=request.query.get("refresh") == "1"
            ),
            headers={"Cache-Control": "no-store"},
        )

    async def conversation_history_handler(request: web.Request) -> web.Response:
        try:
            limit = max(1, min(20000, int(request.query.get("limit", "5000"))))
        except ValueError as error:
            raise web.HTTPBadRequest(text="limit must be an integer") from error
        payload = await dashboard_call(companion().conversation_history, limit)
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def voice_config_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        try:
            segment_seconds = float(body["segment_seconds"])
            rms_threshold = float(body["rms_threshold"])
            voice_model = str(body["voice_model"]).strip()
            voice_name = str(body.get("voice_name", "")).strip() or None
            asr_model = str(body["asr_model"]).strip()
            asr_language = str(body["asr_language"]).strip()
            asr_target_rms = float(body["asr_target_rms"])
            asr_max_gain = float(body["asr_max_gain"])
            vad_input_gain = float(body["vad_input_gain"])
            if (
                not 1 <= segment_seconds <= 15
                or not 0.001 <= rms_threshold <= 1
                or not 0.001 <= asr_target_rms <= 1
                or not 1 <= asr_max_gain <= 48
                or not 1 <= vad_input_gain <= 32
                or re.fullmatch(r"(?:auto|[a-z]{2,3}(?:-[A-Z]{2})?)", asr_language) is None
                or not voice_model or not asr_model
            ):
                raise ValueError("invalid voice settings")
        except (KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(
                text="segment_seconds, rms_threshold, asr_target_rms, asr_max_gain, "
                "vad_input_gain, asr_language, voice_model, and asr_model are required"
            ) from error
        runtime = companion()
        try:
            await runtime.update_voice_config(
                segment_seconds, rms_threshold, voice_model, voice_name, asr_model,
                asr_target_rms, asr_max_gain, vad_input_gain, asr_language,
            )
        except RuntimeError as error:
            raise web.HTTPBadGateway(text=str(error)) from error
        return web.json_response(runtime.telemetry.snapshot(config)["voice"])

    async def voice_action_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        action = str(body.get("action", "")).strip().lower()
        if action != "reconnect":
            raise web.HTTPBadRequest(text="action must be reconnect")
        runtime = companion()
        try:
            await runtime._omnius.ensure_voice_ready(runtime.config.omnius.voice_model)
            await runtime._omnius.configure_supertonic_voice(
                runtime.config.omnius.voice_name
            )
            await runtime._omnius.ensure_asr_model(
                runtime.config.transcription.asr_model
            )
            payload = {"ok": True, "action": action}
        except RuntimeError as error:
            raise web.HTTPBadGateway(text=str(error)) from error
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def action_focus_camera_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        camera_id = str(body.get("camera_id", "")).strip()
        if not camera_id:
            raise web.HTTPBadRequest(text="camera_id is required")
        duration_seconds = body.get("duration_seconds", 45.0)
        try:
            duration_seconds = float(duration_seconds)
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="duration_seconds must be a number") from error
        result = await companion().focus_camera(camera_id, duration_seconds)
        status = 200 if result.get("ok") else 409
        return web.json_response(
            result, status=status, headers={"Cache-Control": "no-store"}
        )

    async def action_inspect_entity_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        entity_id = str(body.get("entity_id", "")).strip()
        if not entity_id:
            raise web.HTTPBadRequest(text="entity_id is required")
        result = await companion().inspect_entity(entity_id)
        status = 200 if result.get("ok") else 409
        return web.json_response(
            result, status=status, headers={"Cache-Control": "no-store"}
        )

    async def memory_handler(_: web.Request) -> web.Response:
        return web.json_response(await dashboard_call(companion().memory_snapshot))

    async def dreams_handler(_: web.Request) -> web.Response:
        return web.json_response(
            await dashboard_call(companion().dreams_snapshot),
            headers={"Cache-Control": "no-store"},
        )

    async def dreams_run_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        try:
            result = await companion().run_identity_dream("manual")
        except RuntimeError as error:
            raise web.HTTPConflict(text=str(error)) from error
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def narratives_handler(request: web.Request) -> web.Response:
        try:
            limit = max(1, min(3650, int(request.query.get("limit", "90"))))
        except ValueError as error:
            raise web.HTTPBadRequest(text="limit must be an integer") from error
        return web.json_response(
            await dashboard_call(companion().daily_narratives, limit),
            headers={"Cache-Control": "no-store"},
        )

    async def narrative_detail_handler(request: web.Request) -> web.Response:
        local_date = request.match_info["local_date"]
        try:
            datetime.strptime(local_date, "%Y-%m-%d")
        except ValueError as error:
            raise web.HTTPBadRequest(text="local_date must be YYYY-MM-DD") from error
        result = await dashboard_call(companion().daily_narrative, local_date)
        if result is None:
            raise web.HTTPNotFound(text="daily narrative not found")
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def memory_entity_handler(request: web.Request) -> web.Response:
        detail = await dashboard_call(
            companion().inspect_memory_entity, request.match_info["entity_id"]
        )
        if detail is None:
            raise web.HTTPNotFound(text="memory entity not found")
        return web.json_response(detail)

    async def memory_episodes_handler(_: web.Request) -> web.Response:
        return web.json_response(await dashboard_call(companion().memory_episodes))

    async def memory_claims_handler(_: web.Request) -> web.Response:
        return web.json_response(await dashboard_call(companion().memory_claims))

    async def memory_alias_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        try:
            result = await dashboard_call(
                companion().add_memory_alias, request.match_info["entity_id"], str(body["alias"])
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="memory entity not found") from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(result)

    async def memory_correct_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        try:
            result = await dashboard_call(
                companion().correct_memory_claim,
                request.match_info["claim_id"],
                str(body["replacement"]),
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="memory claim not found") from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(result)

    async def memory_export_handler(_: web.Request) -> web.Response:
        try:
            payload = await dashboard_call(companion().export_memory)
        except PermissionError as error:
            raise web.HTTPForbidden(text=str(error)) from error
        return web.json_response(
            payload,
            headers={"Content-Disposition": "attachment; filename=egg-memory-export.json"},
        )

    async def memory_entity_export_handler(request: web.Request) -> web.Response:
        try:
            payload = await dashboard_call(
                companion().export_memory_entity, request.match_info["entity_id"]
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="memory entity not found") from error
        except PermissionError as error:
            raise web.HTTPForbidden(text=str(error)) from error
        return web.json_response(payload)

    async def memory_revision_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        try:
            result = await dashboard_call(
                companion().revise_memory,
                str(body["target_type"]),
                str(body["target_id"]),
                str(body["decision"]),
                str(body["replacement"]) if body.get("replacement") is not None else None,
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="revision target not found") from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(result)

    async def cognition_state_handler(_: web.Request) -> web.Response:
        snapshot = companion().telemetry.snapshot(config)
        return web.json_response(
            {
                "attention": snapshot["attention_decisions"],
                "interactions": snapshot["interaction_decisions"],
                "retrieval": snapshot.get("retrieval_hits", []),
                "consolidation": snapshot["consolidation"],
                "memory": snapshot["memory"],
            }
        )

    async def memory_delete_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        try:
            await dashboard_call(
                companion().delete_memory_entity, request.match_info["entity_id"]
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="memory entity not found") from error
        except PermissionError as error:
            raise web.HTTPForbidden(text=str(error)) from error
        return web.json_response({"deleted": request.match_info["entity_id"]})

    async def memory_consolidate_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        runtime = companion()
        if runtime._memory is None:
            raise web.HTTPConflict(text="cognitive memory is disabled")
        result = await dashboard_call(runtime._memory.consolidate)
        runtime.telemetry.record_consolidation(result)
        return web.json_response(result)

    async def world_summary_handler(_: web.Request) -> web.Response:
        pipeline = companion()._memory
        query = getattr(pipeline, "world_query", None) if pipeline else None
        if query is None:
            return web.json_response({"error": "world model not initialized"}, status=503)
        return web.json_response(
            await dashboard_call(query.world_summary),
            headers={"Cache-Control": "no-store"},
        )

    async def occupancy_snapshot_handler(_: web.Request) -> web.Response:
        return web.json_response(
            companion().occupancy_snapshot(), headers={"Cache-Control": "no-store"}
        )

    async def occupancy_resolution_handler(request: web.Request) -> web.Response:
        require_loopback(request)
        body = await request.json()
        try:
            sample_stride = int(body["sample_stride"])
        except (KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="sample_stride is required") from error
        applied = companion().update_occupancy_resolution(sample_stride)
        return web.json_response(applied)

    async def world_entity_handler(request: web.Request) -> web.Response:
        entity_id = request.match_info["entity_id"]
        pipeline = companion()._memory
        query = getattr(pipeline, "world_query", None) if pipeline else None
        if query is None:
            return web.json_response({"error": "world model not initialized"}, status=503)
        result = await dashboard_call(query.explain_entity, entity_id)
        if result is None:
            return web.json_response({"error": "entity not found"}, status=404)
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def world_conflicts_handler(_: web.Request) -> web.Response:
        pipeline = companion()._memory
        query = getattr(pipeline, "world_query", None) if pipeline else None
        if query is None:
            return web.json_response({"error": "world model not initialized"}, status=503)
        conflicts = await dashboard_call(query.conflicts)
        return web.json_response(
            [{"entity_id": c.entity_id, "property_id": c.property_id,
              "current_value": c.current_value, "proposed_value": c.proposed_value,
              "reason": c.reason, "assertions": c.assertions} for c in conflicts],
            headers={"Cache-Control": "no-store"},
        )

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/state", state_handler)
    app.router.add_get("/api/config", config_handler)
    app.router.add_get("/api/graph", graph_handler)
    app.router.add_get("/api/graph/node", graph_node_handler)
    app.router.add_get("/api/cameras/{camera_id}/raw.jpg", raw_frame_handler)
    app.router.add_get("/api/cameras/{camera_id}/stream.mjpg", camera_stream_handler)
    app.router.add_get("/api/audio/stream", audio_stream_handler)
    app.router.add_get("/api/identities/{profile_id}/face.jpg", identity_handler)
    app.router.add_get("/api/identities/{profile_id}/timeline", identity_timeline_handler)
    app.router.add_get(
        "/api/identities/{profile_id}/samples/{sample_id}.jpg", identity_sample_handler
    )
    app.router.add_get("/api/objects/{profile_id}/mask.png", object_handler)
    app.router.add_get("/api/voice/catalog", voice_catalog_handler)
    app.router.add_get("/api/voice/conversation", conversation_history_handler)
    app.router.add_put("/api/voice/config", voice_config_handler)
    app.router.add_post("/api/voice/action", voice_action_handler)
    app.router.add_post("/api/actions/focus_camera", action_focus_camera_handler)
    app.router.add_post("/api/actions/inspect_entity", action_inspect_entity_handler)
    app.router.add_get("/api/dreams", dreams_handler)
    app.router.add_post("/api/dreams/run", dreams_run_handler)
    app.router.add_get("/api/memory/narratives", narratives_handler)
    app.router.add_get(
        "/api/memory/narratives/{local_date}", narrative_detail_handler
    )
    app.router.add_get("/api/memory", memory_handler)
    app.router.add_get("/api/memory/episodes", memory_episodes_handler)
    app.router.add_get("/api/memory/claims", memory_claims_handler)
    app.router.add_get("/api/memory/export", memory_export_handler)
    app.router.add_get("/api/memory/export/{entity_id}", memory_entity_export_handler)
    app.router.add_get("/api/memory/entities/{entity_id}", memory_entity_handler)
    app.router.add_get("/api/memory/evidence/{evidence_id}/media", evidence_media_handler)
    app.router.add_post("/api/memory/entities/{entity_id}/aliases", memory_alias_handler)
    app.router.add_post("/api/memory/entities/{entity_id}/alias", memory_alias_handler)
    app.router.add_delete("/api/memory/entities/{entity_id}", memory_delete_handler)
    app.router.add_post("/api/memory/claims/{claim_id}/correct", memory_correct_handler)
    app.router.add_post("/api/memory/revisions", memory_revision_handler)
    app.router.add_post("/api/memory/consolidate", memory_consolidate_handler)
    app.router.add_get("/api/cognition/state", cognition_state_handler)
    app.router.add_get("/api/world", world_summary_handler)
    app.router.add_get("/api/occupancy", occupancy_snapshot_handler)
    app.router.add_put("/api/occupancy/resolution", occupancy_resolution_handler)
    app.router.add_get("/api/world/entity/{entity_id}", world_entity_handler)
    app.router.add_get("/api/world/conflicts", world_conflicts_handler)
    app.router.add_static(
        "/assets/",
        Path(__file__).with_name("vendor"),
        name="dashboard-assets",
        show_index=False,
        append_version=False,
    )
    app.router.add_get("/{route:.*}", index_handler)
    await refresh()
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"Egg dashboard: http://127.0.0.1:{port}")
    try:
        await asyncio.Event().wait()
    finally:
        task = state["runtime_task"]
        if isinstance(task, asyncio.Task):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        audit_task = readiness._task
        if isinstance(audit_task, asyncio.Task):
            audit_task.cancel()
            await asyncio.gather(audit_task, return_exceptions=True)
        await runner.cleanup()
        dashboard_executor.shutdown(wait=False, cancel_futures=True)
