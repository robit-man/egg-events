import * as THREE from 'three';
import { OrbitControls } from '/assets/OrbitControls.js?v=20260811a';

function initCanvasFallback(container) {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', 'Interactive compatibility rendering of the multimodal knowledge graph');
  container.replaceChildren(canvas);
  const context = canvas.getContext('2d', { alpha: false });
  const colors = { person:'#60a5fa', appearance:'#38bdf8', object:'#34d399', object_category:'#2dd4bf', sound_event:'#ffae00', content:'#ffae00', evidence:'#c084fc', claim:'#fb7185', episode:'#94a3b8', entity:'#22d3ee' };
  let data = { nodes: [], links: [] }, nodes = [], links = [], selected = null, hovered = null;
  let query = '', kind = '', pixelRatio = 1, width = 1, height = 1;
  let zoom = 1, panX = 0, panY = 0, yaw = 0.22, pitch = -0.12;
  let target = { x:0, y:0, z:0 }, drag = null, moved = false, viewInitialized = false;
  let lastDreamRevision = '';
  let lastActivationSequence = 0;
  const appendFlashMs = 1600;
  const activationPulseMs = 1250;
  const activationHopMs = 150;
  const layoutTweenMs = 1250;

  function hash(value) {
    let result = 2166136261;
    for (const character of String(value)) { result ^= character.charCodeAt(0); result = Math.imul(result, 16777619); }
    return result >>> 0;
  }
  function color(node) {
    const subtype = String(node.subtype || '').toLowerCase();
    if (subtype.includes('person') || subtype.includes('face')) return colors.person;
    if (subtype.includes('appearance')) return colors.appearance;
    if (subtype.includes('sound')) return colors.sound_event;
    if (subtype.includes('content') || subtype.includes('ocr')) return colors.content;
    if (subtype.includes('object')) return colors.object;
    return colors[node.kind] || colors.entity;
  }
  function firingColor(source) {
    if (source === 'voice') return '#ffae00';
    if (source === 'memory_recall') return '#c084fc';
    if (source === 'action') return '#34d399';
    return '#ffffff';
  }
  function matches(node) {
    const text = `${node.label || ''} ${node.source_id || ''} ${node.subtype || ''}`.toLowerCase();
    return (!kind || node.kind === kind) && (!query || text.includes(query));
  }
  function layout(payload) {
    const selectedId = selected?.id || null;
    const hadGraph = nodes.length > 0;
    const previousPositions = new Map(nodes.map(node => [node.id, {x:node.x,y:node.y,z:node.z||0}]));
    const previousNodeIds = new Set(nodes.map(node => node.id));
    const previousLinkIds = new Set(links.map(link => linkIdentity(link)));
    const appendedAt = performance.now();
    data = payload || { nodes: [], links: [] };
    const dreamRevision = String(data.dream?.revision || '');
    const dreamChanged = hadGraph && Boolean(dreamRevision) && dreamRevision !== lastDreamRevision;
    lastDreamRevision = dreamRevision;
    const dreamTouched = new Set(data.dream?.touched_node_ids || []);
    if (dreamChanged) for (const link of data.links || []) if (dreamTouched.has(link.source) || dreamTouched.has(link.target)) { dreamTouched.add(link.source); dreamTouched.add(link.target); }
    const timestamps = (data.nodes || []).map(node => Date.parse(node.updated_at || '')).filter(Number.isFinite);
    const oldest = timestamps.length ? Math.min(...timestamps) : 0, newest = timestamps.length ? Math.max(...timestamps) : 0;
    const degree = new Map();
    for (const link of data.links || []) {
      degree.set(link.source, (degree.get(link.source) || 0) + 1);
      degree.set(link.target, (degree.get(link.target) || 0) + 1);
    }
    nodes = (data.nodes || []).map((node, index) => {
      const seed = hash(node.id), angle = ((seed % 10000) / 10000) * Math.PI * 2;
      const subtype = String(node.subtype || '').toLowerCase();
      const person = subtype.includes('person') || subtype.includes('face') || subtype.includes('appearance');
      const content = subtype.includes('content') || subtype.includes('ocr');
      const lobe = person ? -1 : content ? 1 : (seed & 1 ? 1 : -1);
      const radius = 7 + ((seed >>> 8) % 1300) / 100;
      const shell = node.kind === 'evidence' ? 1.24 : node.kind === 'episode' ? 0.7 : 1;
      const timestamp = Date.parse(node.updated_at || ''), chronology = newest > oldest && Number.isFinite(timestamp) ? ((timestamp-oldest)/(newest-oldest)-.5)*30 : 0;
      return { ...node, x:lobe * (7 + radius * .42) + Math.cos(angle) * radius * .7 * shell, y:Math.sin(angle) * radius * .64 * shell, z:chronology + ((((seed >>> 16) % 1000) / 1000) - .5) * 4 * shell, degree:degree.get(node.id) || 0, index, appendedAt:hadGraph && (!previousNodeIds.has(node.id) || (dreamChanged && dreamTouched.has(node.id))) ? appendedAt : 0 };
    });
    const byId = new Map(nodes.map(node => [node.id, node]));
    links = (data.links || []).filter(link => byId.has(link.source) && byId.has(link.target)).map(link => ({ ...link, sourceNode:byId.get(link.source), targetNode:byId.get(link.target), appendedAt:hadGraph && !previousLinkIds.has(linkIdentity(link)) ? appendedAt : 0 }));
    for (let iteration = 0; iteration < 34; iteration += 1) {
      const temperature = .16 * (1 - iteration / 40);
      for (const link of links) {
        const dx = link.targetNode.x - link.sourceNode.x, dy = link.targetNode.y - link.sourceNode.y;
        const distance = Math.max(.05, Math.hypot(dx, dy)), confidence = Math.max(.04, Math.min(1, Number(link.confidence ?? .5)));
        const desired = 4.8 + (1 - confidence) * 14, force = (distance - desired) * temperature * (.25 + confidence * .55) / distance;
        link.sourceNode.x += dx * force * .5; link.sourceNode.y += dy * force * .5;
        link.targetNode.x -= dx * force * .5; link.targetNode.y -= dy * force * .5;
      }
      for (let index = 0; index < nodes.length; index += 1) {
        const node = nodes[index];
        for (let sample = 1; sample <= Math.min(7, nodes.length - 1); sample += 1) {
          const other = nodes[(index + sample * 97 + iteration * 31) % nodes.length];
          if (other === node) continue;
          const dx = node.x - other.x, dy = node.y - other.y, squared = Math.max(.4, dx * dx + dy * dy);
          if (squared < 30) { node.x += dx * temperature * 1.4 / squared; node.y += dy * temperature * 1.4 / squared; }
        }
      }
    }
    for (const node of nodes) {
      const prior = previousPositions.get(node.id);
      node.tx=node.x;node.ty=node.y;node.tz=node.z||0;
      node.fx=prior?.x??node.x;node.fy=prior?.y??node.y;node.fz=prior?.z??(node.z||0);
      if (hadGraph && prior) { node.x=node.fx;node.y=node.fy;node.z=node.fz;node.tweenAt=appendedAt; }
    }
    selected = selectedId ? nodes.find(node => node.id === selectedId) || null : null;
    if (!viewInitialized) reset();
  }
  function linkIdentity(link) { return String(link.id || `${link.source}:${link.relation || ''}:${link.target}`); }
  function appendFlash(item) { return item.appendedAt ? Math.max(0, 1 - (performance.now() - item.appendedAt) / appendFlashMs) : 0; }
  function activationPulse(item, now = performance.now()) {
    if (!item.activationAt) return 0;
    const elapsed = now - item.activationAt;
    if (elapsed < 0 || elapsed >= activationPulseMs) return 0;
    const progress = elapsed / activationPulseMs;
    return Math.sin(progress * Math.PI) * Number(item.activationIntensity || 1);
  }
  function fireActivations(payload) {
    const events = Array.isArray(payload?.events) ? payload.events : [];
    const byId = new Map(nodes.map(node => [node.id, node]));
    const adjacency = new Map();
    for (const link of links) {
      if (!adjacency.has(link.source)) adjacency.set(link.source, []);
      if (!adjacency.has(link.target)) adjacency.set(link.target, []);
      adjacency.get(link.source).push({nodeId:link.target,link});
      adjacency.get(link.target).push({nodeId:link.source,link});
    }
    for (const event of events.sort((left, right) => Number(left.sequence || 0) - Number(right.sequence || 0))) {
      const sequence = Number(event.sequence || 0);
      if (!sequence || sequence <= lastActivationSequence) continue;
      lastActivationSequence = sequence;
      const now = performance.now(), intensity = Math.max(.1, Math.min(1, Number(event.intensity || 1)));
      const origins = (event.origin_node_ids || []).filter(nodeId => byId.has(nodeId));
      const explicit = (event.node_ids || []).filter(nodeId => byId.has(nodeId));
      const seeds = origins.length ? origins : explicit;
      const schedule = new Map(seeds.map(nodeId => [nodeId, 0]));
      for (const nodeId of explicit) if (!schedule.has(nodeId)) schedule.set(nodeId, activationHopMs);
      const queue = [...schedule].map(([nodeId, delay]) => ({nodeId,delay,depth:delay ? 1 : 0}));
      let traversed = 0;
      while (queue.length && traversed < 120) {
        const current = queue.shift(); traversed += 1;
        if (!event.cascade || current.depth >= 3) continue;
        const neighbors = [...(adjacency.get(current.nodeId) || [])]
          .sort((left, right) => Number(right.link.confidence || 0) - Number(left.link.confidence || 0)).slice(0, 10);
        for (const neighbor of neighbors) {
          const delay = current.delay + activationHopMs;
          neighbor.link.activationAt = now + current.delay + activationHopMs * .45;
          neighbor.link.activationIntensity = intensity * Math.pow(.78, current.depth);
          neighbor.link.activationFrom = current.nodeId;
          neighbor.link.activationColor = firingColor(event.source);
          if (!schedule.has(neighbor.nodeId) || delay < schedule.get(neighbor.nodeId)) {
            schedule.set(neighbor.nodeId, delay);
            queue.push({nodeId:neighbor.nodeId,delay,depth:current.depth+1});
          }
        }
      }
      for (const [nodeId, delay] of schedule) {
        const node = byId.get(nodeId);
        node.activationAt = now + delay;
        node.activationIntensity = intensity * Math.pow(.82, Math.round(delay / activationHopMs));
        node.activationSource = event.source;
        node.activationColor = firingColor(event.source);
      }
    }
  }
  function graphBounds() {
    if (!nodes.length) return { minX:-25, maxX:25, minY:-20, maxY:20, minZ:-10, maxZ:10 };
    return nodes.reduce((bounds, node) => ({ minX:Math.min(bounds.minX,node.x), maxX:Math.max(bounds.maxX,node.x), minY:Math.min(bounds.minY,node.y), maxY:Math.max(bounds.maxY,node.y), minZ:Math.min(bounds.minZ,node.z||0), maxZ:Math.max(bounds.maxZ,node.z||0) }), { minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity,minZ:Infinity,maxZ:-Infinity });
  }
  function reset() {
    const bounds = graphBounds(), spanX = Math.max(10, bounds.maxX - bounds.minX), spanY = Math.max(10, bounds.maxY - bounds.minY);
    zoom = Math.max(3, Math.min((width - 50) / spanX, (height - 50) / spanY));
    target = { x:(bounds.minX + bounds.maxX) / 2, y:(bounds.minY + bounds.maxY) / 2, z:(bounds.minZ + bounds.maxZ) / 2 };
    panX = 0; panY = 0; yaw = .22; pitch = -.12; viewInitialized = true;
  }
  function point(node) {
    const x=node.x-target.x, y=node.y-target.y, z=(node.z||0)-target.z;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const rx=cy*x+sy*z, rz=-sy*x+cy*z, ry=cp*y-sp*rz, depth=sp*y+cp*rz;
    const perspective=Math.max(.42,Math.min(1.8,1-depth/95));
    return { x:width/2+panX+rx*zoom*perspective, y:height/2+panY+ry*zoom*perspective, depth, scale:perspective };
  }
  function radius(node) { return 2.1 + Math.min(4.5, Math.log2(node.degree + 1) * .7) + Number(node.confidence ?? .5) * .7; }
  function render() {
    const now = performance.now();
    for (const node of nodes) if (node.tweenAt) { const raw=Math.min(1,(now-node.tweenAt)/layoutTweenMs), eased=1-Math.pow(1-raw,3); node.x=node.fx+(node.tx-node.fx)*eased;node.y=node.fy+(node.ty-node.fy)*eased;node.z=node.fz+(node.tz-node.fz)*eased;if(raw>=1)node.tweenAt=0; }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = '#070d19'; context.fillRect(0, 0, width, height);
    for (const link of links) {
      const source = point(link.sourceNode), target = point(link.targetNode), confidence = Math.max(.05, Math.min(1, Number(link.confidence ?? .5)));
      const visible = matches(link.sourceNode) && matches(link.targetNode), dx = target.x - source.x, dy = target.y - source.y, length = Math.max(1, Math.hypot(dx,dy));
      const bend = ((hash(link.id || `${link.source}:${link.target}`) & 1) ? 1 : -1) * (7 + (1-confidence) * 15);
      context.beginPath(); context.moveTo(source.x, source.y); context.quadraticCurveTo((source.x+target.x)/2-dy/length*bend, (source.y+target.y)/2+dx/length*bend, target.x, target.y);
      context.strokeStyle = visible ? `rgba(102,126,168,${.16 + confidence * .36})` : 'rgba(68,82,108,.025)';
      context.lineWidth = visible ? .45 + confidence * 2.4 + Math.min(1.6, Math.log2(Math.max(1,Number(link.confirmations || 1))) * .35) : .35;
      context.stroke();
      const flash = appendFlash(link);
      if (flash > 0 && visible) { context.strokeStyle = `rgba(255,255,255,${flash})`; context.lineWidth += 1.8 * flash; context.stroke(); }
      const firing = activationPulse(link, now);
      if (firing > 0 && visible) {
        context.globalAlpha = Math.min(1,.22+firing); context.strokeStyle = link.activationColor || '#fff'; context.lineWidth += 2.6 * firing; context.stroke();context.globalAlpha=1;
        const raw = Math.max(0, Math.min(1, (now-link.activationAt)/activationPulseMs));
        const t = link.activationFrom === link.target ? 1-raw : raw, inverse = 1-t;
        const controlX=(source.x+target.x)/2-dy/length*bend, controlY=(source.y+target.y)/2+dx/length*bend;
        const sparkX=inverse*inverse*source.x+2*inverse*t*controlX+t*t*target.x, sparkY=inverse*inverse*source.y+2*inverse*t*controlY+t*t*target.y;
        context.fillStyle=link.activationColor||'#fff';context.shadowColor=link.activationColor||'#fff';context.shadowBlur=14*firing;context.beginPath();context.arc(sparkX,sparkY,1.8+2.4*firing,0,Math.PI*2);context.fill();context.shadowBlur=0;
      }
    }
    for (const node of nodes) {
      const screen = point(node), visible = matches(node), size = radius(node) * screen.scale * (selected === node ? 1.65 : hovered === node ? 1.3 : 1);
      context.globalAlpha = visible ? 1 : .08; context.fillStyle = color(node); context.shadowColor = color(node); context.shadowBlur = visible ? 7 : 0;
      context.beginPath(); context.arc(screen.x, screen.y, size, 0, Math.PI * 2); context.fill();
      const flash = appendFlash(node);
      if (flash > 0 && visible) { context.globalAlpha = flash; context.fillStyle = '#fff'; context.shadowColor = '#fff'; context.shadowBlur = 18 * flash; context.beginPath(); context.arc(screen.x, screen.y, size * (1 + flash * .22), 0, Math.PI * 2); context.fill(); }
      const firing = activationPulse(node, now);
      if (firing > 0 && visible) { context.globalAlpha = Math.min(1,.3+firing); context.fillStyle = node.activationColor || '#fff'; context.shadowColor = node.activationColor || '#fff'; context.shadowBlur = 24 * firing; context.beginPath(); context.arc(screen.x, screen.y, size * (1 + firing * .42), 0, Math.PI * 2); context.fill(); }
    }
    context.shadowBlur = 0; context.globalAlpha = 1;
    const focus = hovered || selected;
    if (focus) {
      const screen = point(focus), label = String(focus.label || focus.source_id || focus.id).slice(0, 64);
      context.font = '600 12px ui-monospace, SFMono-Regular, Consolas, monospace';
      const labelWidth = Math.min(width - 24, context.measureText(label).width + 22), x = Math.max(8, Math.min(width-labelWidth-8,screen.x-labelWidth/2)), y = Math.max(26,screen.y-radius(focus)-19);
      context.fillStyle = 'rgba(16,24,40,.96)'; context.fillRect(x,y-20,labelWidth,28); context.strokeStyle = '#ffae00'; context.lineWidth = 1; context.strokeRect(x+.5,y-19.5,labelWidth-1,27); context.fillStyle = '#f2f4f7'; context.fillText(label,x+11,y-2);
    }
    setTimeout(() => requestAnimationFrame(render), 45);
  }
  function select(node) {
    selected = node;
    if (!node) return;
    target = {x:node.x,y:node.y,z:node.z||0}; panX=0;panY=0;
    const neighbors = links.filter(link => link.source === node.id || link.target === node.id).map(link => ({ ...link, label:(link.source === node.id ? link.targetNode : link.sourceNode).label || (link.source === node.id ? link.target : link.source) }));
    window.dispatchEvent(new CustomEvent('egg:graph-selection', { detail:{ node, neighbors } }));
  }
  function hitTest(event) {
    const bounds = canvas.getBoundingClientRect(), x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    let nearest = null, nearestDistance = 12;
    for (const node of nodes) { const screen = point(node), distance = Math.hypot(screen.x-x,screen.y-y); if (distance < nearestDistance) { nearest=node; nearestDistance=distance; } }
    return nearest;
  }
  canvas.addEventListener('pointerdown', event => { canvas.setPointerCapture(event.pointerId); drag={x:event.clientX,y:event.clientY,panX,panY,yaw,pitch,pan:event.shiftKey||event.button!==0}; moved=false; });
  canvas.addEventListener('pointermove', event => { hovered=hitTest(event); canvas.style.cursor=hovered?'pointer':'grab'; if (!drag) return; const dx=event.clientX-drag.x,dy=event.clientY-drag.y; if(Math.hypot(dx,dy)>3)moved=true; if(drag.pan){panX=drag.panX+dx;panY=drag.panY+dy;}else{yaw=drag.yaw-dx*.007;pitch=Math.max(-1.35,Math.min(1.35,drag.pitch+dy*.007));} });
  canvas.addEventListener('pointerup', event => { if(!moved)select(hitTest(event));drag=null; });
  canvas.addEventListener('pointercancel', () => { drag=null; });
  canvas.addEventListener('contextmenu', event => event.preventDefault());
  canvas.addEventListener('wheel', event => { event.preventDefault(); const bounds=canvas.getBoundingClientRect(), mx=event.clientX-bounds.left-width/2, my=event.clientY-bounds.top-height/2, factor=Math.exp(-event.deltaY*.001); panX=mx-(mx-panX)*factor;panY=my-(my-panY)*factor;zoom=Math.max(.5,Math.min(80,zoom*factor)); }, {passive:false});
  new ResizeObserver(() => { const wasRenderable=width>50&&height>50;width=Math.max(1,container.clientWidth);height=Math.max(1,container.clientHeight);pixelRatio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(width*pixelRatio);canvas.height=Math.round(height*pixelRatio);canvas.style.width=`${width}px`;canvas.style.height=`${height}px`; const becameRenderable=!wasRenderable&&width>50&&height>50;if(nodes.length&&(!viewInitialized||becameRenderable))reset(); }).observe(container);
  window.addEventListener('egg:graph-data', event => layout(event.detail));
  window.addEventListener('egg:graph-activations', event => fireActivations(event.detail));
  window.addEventListener('egg:graph-filter', event => { query=String(event.detail?.query||'').trim().toLowerCase();kind=String(event.detail?.kind||''); });
  window.addEventListener('egg:graph-reset', reset);
  if (window.__eggGraphData) layout(window.__eggGraphData);
  if (window.__eggGraphActivations) fireActivations(window.__eggGraphActivations);
  window.dispatchEvent(new CustomEvent('egg:graph-renderer', {detail:{mode:'canvas-3d'}}));
  render();
}

const container = document.getElementById('knowledge-graph');

if (container) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    const context = renderer.getContext();
    const debug = context.getExtension('WEBGL_debug_renderer_info');
    const implementation = String(debug ? context.getParameter(debug.UNMASKED_RENDERER_WEBGL) : '');
    if (/swiftshader|llvmpipe|software/i.test(implementation)) {
      renderer.dispose();
      renderer = undefined;
      initCanvasFallback(container);
    }
  } catch (error) {
    initCanvasFallback(container);
  }

  if (renderer) {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x070d19, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x070d19, 0.018);
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 500);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.minDistance = 8;
    controls.maxDistance = 145;
    controls.zoomToCursor = true;
    controls.screenSpacePanning = true;
    controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;

    scene.add(new THREE.HemisphereLight(0xbcd7ff, 0x08101d, 1.55));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    keyLight.position.set(16, 24, 20);
    scene.add(keyLight);
    const rimLight = new THREE.PointLight(0x567cff, 42, 100);
    rimLight.position.set(-22, 4, -18);
    scene.add(rimLight);

    const graphRoot = new THREE.Group();
    scene.add(graphRoot);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(2, 2);
    const meshes = [];
    const meshById = new Map();
    const nodeById = new Map();
    const linksByNode = new Map();
    const edgeObjects = [];
    const edgeById = new Map();
    let graphData = { nodes: [], links: [] };
    let hovered = null;
    let selected = null;
    let labelSprite = null;
    let pointerDown = null;
    let appendPulseObjects = [];
    const activePulseObjects = new Set();
    let lastDreamRevision = '';
    let lastActivationSequence = 0;
    const appendFlashMs = 1600;
    const activationPulseMs = 1250;
    const activationHopMs = 150;
    const layoutTweenMs = 1250;
    const white = new THREE.Color(0xffffff);

    const colors = {
      person: 0x60a5fa,
      appearance: 0x38bdf8,
      object: 0x34d399,
      object_category: 0x2dd4bf,
      sound_event: 0xffae00,
      content: 0xfbbf24,
      evidence: 0xc084fc,
      claim: 0xfb7185,
      episode: 0x94a3b8,
      entity: 0x22d3ee,
    };

    function hash(value) {
      let result = 2166136261;
      for (const character of String(value)) {
        result ^= character.charCodeAt(0);
        result = Math.imul(result, 16777619);
      }
      return result >>> 0;
    }

    function randomUnit(seed) {
      let value = seed || 1;
      return () => {
        value += 0x6d2b79f5;
        let next = value;
        next = Math.imul(next ^ (next >>> 15), next | 1);
        next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
        return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
      };
    }

    function nodeColor(node) {
      const subtype = String(node.subtype || '').toLowerCase();
      if (subtype.includes('person') || subtype.includes('face')) return colors.person;
      if (subtype.includes('appearance')) return colors.appearance;
      if (subtype.includes('sound')) return colors.sound_event;
      if (subtype.includes('content') || subtype.includes('ocr')) return colors.content;
      if (subtype.includes('object')) return colors.object;
      return colors[node.kind] || colors.entity;
    }

    function firingColor(source) {
      if (source === 'voice') return 0xffae00;
      if (source === 'memory_recall') return 0xc084fc;
      if (source === 'action') return 0x34d399;
      return 0xffffff;
    }

    function initialPosition(node, index, total) {
      const seed = hash(node.id);
      const random = randomUnit(seed);
      const angle = random() * Math.PI * 2;
      const radius = 8 + random() * 14;
      const subtype = String(node.subtype || '').toLowerCase();
      const content = subtype.includes('content') || subtype.includes('ocr');
      const person = subtype.includes('person') || subtype.includes('face') || subtype.includes('appearance');
      const lobe = person ? -1 : content ? 1 : (seed & 1 ? 1 : -1);
      const shell = node.kind === 'evidence' ? 1.25 : node.kind === 'episode' ? 0.62 : 1;
      return new THREE.Vector3(
        lobe * (7 + radius * 0.46) + Math.cos(angle) * radius * 0.7 * shell,
        Math.sin(angle) * radius * 0.64 * shell,
        (random() - 0.5) * 20 * shell + Math.sin((index / Math.max(1, total)) * Math.PI * 4) * 2,
      );
    }

    function relationshipLayout(nodes, links) {
      const positions = new Map(nodes.map((node, index) => [node.id, initialPosition(node, index, nodes.length)]));
      const timestamps = nodes.map(node => Date.parse(node.updated_at || '')).filter(Number.isFinite);
      const oldest = timestamps.length ? Math.min(...timestamps) : 0;
      const newest = timestamps.length ? Math.max(...timestamps) : 0;
      const chronology = new Map();
      for (const node of nodes) {
        const timestamp = Date.parse(node.updated_at || '');
        const temporalZ = newest > oldest && Number.isFinite(timestamp) ? ((timestamp - oldest) / (newest - oldest) - 0.5) * 30 : 0;
        chronology.set(node.id, temporalZ);
        positions.get(node.id).z = temporalZ + positions.get(node.id).z * 0.2;
      }
      const usableLinks = links.filter(link => positions.has(link.source) && positions.has(link.target));
      const scratch = new THREE.Vector3();
      for (let iteration = 0; iteration < 42; iteration += 1) {
        const temperature = 0.21 * (1 - iteration / 50);
        for (const link of usableLinks) {
          const source = positions.get(link.source), target = positions.get(link.target);
          scratch.subVectors(target, source);
          const distance = Math.max(0.01, scratch.length());
          const confidence = THREE.MathUtils.clamp(Number(link.confidence ?? 0.5), 0.04, 1);
          const desired = 5.2 + (1 - confidence) * 15;
          const adjustment = (distance - desired) * temperature * (0.25 + confidence * 0.55);
          scratch.multiplyScalar(adjustment / distance);
          source.addScaledVector(scratch, 0.5);
          target.addScaledVector(scratch, -0.5);
        }
        for (let index = 0; index < nodes.length; index += 1) {
          const node = nodes[index], position = positions.get(node.id), random = randomUnit(hash(node.id) + iteration);
          for (let sample = 0; sample < Math.min(9, nodes.length - 1); sample += 1) {
            const other = positions.get(nodes[Math.floor(random() * nodes.length)].id);
            if (other === position) continue;
            scratch.subVectors(position, other);
            const squared = Math.max(0.45, scratch.lengthSq());
            if (squared < 36) position.addScaledVector(scratch, temperature * 1.8 / squared);
          }
          position.multiplyScalar(0.9985);
          position.z += (chronology.get(node.id) - position.z) * temperature * 0.045;
        }
      }
      return positions;
    }

    function disposeObject(object) {
      object.traverse(child => {
        child.geometry?.dispose?.();
        if (Array.isArray(child.material)) child.material.forEach(material => material.dispose());
        else child.material?.dispose?.();
      });
    }

    function clearGraph() {
      if (labelSprite) {
        labelSprite.material.map?.dispose();
        labelSprite.material.dispose();
        graphRoot.remove(labelSprite);
        labelSprite = null;
      }
      for (const child of [...graphRoot.children]) {
        graphRoot.remove(child);
        disposeObject(child);
      }
      meshes.length = 0;
      edgeObjects.length = 0;
      appendPulseObjects = [];
      activePulseObjects.clear();
      meshById.clear();
      nodeById.clear();
      linksByNode.clear();
      edgeById.clear();
      hovered = null;
      selected = null;
    }

    function curveFor(source, target, link) {
      const midpoint = source.clone().add(target).multiplyScalar(0.5);
      const chord = target.clone().sub(source);
      const normal = new THREE.Vector3(-chord.y, chord.x, chord.z * 0.18);
      if (normal.lengthSq() < 0.001) normal.set(0, 1, 0);
      normal.normalize();
      const confidence = THREE.MathUtils.clamp(Number(link.confidence ?? 0.5), 0, 1);
      const direction = (hash(link.id || `${link.source}:${link.target}`) & 1) ? 1 : -1;
      midpoint.addScaledVector(normal, direction * (0.65 + (1 - confidence) * 1.7));
      return new THREE.CatmullRomCurve3([source, midpoint, target]);
    }

    function linkIdentity(link) {
      return String(link.id || `${link.source}:${link.relation || ''}:${link.target}`);
    }

    function buildGraph(payload) {
      const selectedId = selected?.userData?.node?.id || null;
      const hadGraph = meshes.length > 0;
      const previousPositions = new Map(meshes.map(mesh => [mesh.userData.node.id, mesh.position.clone()]));
      const previousNodeIds = new Set((graphData.nodes || []).map(node => node.id));
      const previousLinkIds = new Set((graphData.links || []).map(link => linkIdentity(link)));
      const appendedAt = performance.now();
      const savedCamera = camera.position.clone();
      const savedTarget = controls.target.clone();
      graphData = payload || { nodes: [], links: [] };
      const dreamRevision = String(graphData.dream?.revision || '');
      const dreamChanged = hadGraph && Boolean(dreamRevision) && dreamRevision !== lastDreamRevision;
      lastDreamRevision = dreamRevision;
      clearGraph();
      const nodes = Array.isArray(graphData.nodes) ? graphData.nodes : [];
      const links = Array.isArray(graphData.links) ? graphData.links : [];
      const dreamTouched = new Set(graphData.dream?.touched_node_ids || []);
      if (dreamChanged) for (const link of links) if (dreamTouched.has(link.source) || dreamTouched.has(link.target)) { dreamTouched.add(link.source); dreamTouched.add(link.target); }
      const positions = relationshipLayout(nodes, links);
      nodes.forEach(node => nodeById.set(node.id, node));
      for (const link of links) {
        if (!linksByNode.has(link.source)) linksByNode.set(link.source, []);
        if (!linksByNode.has(link.target)) linksByNode.set(link.target, []);
        linksByNode.get(link.source).push(link);
        linksByNode.get(link.target).push(link);
      }

      const renderedLinks = links
        .filter(link => positions.has(link.source) && positions.has(link.target))
        .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))
        .slice(0, 1800);
      for (const link of renderedLinks) {
        const source = positions.get(link.source), target = positions.get(link.target);
        const confidence = THREE.MathUtils.clamp(Number(link.confidence ?? 0.5), 0.05, 1);
        const confirmations = Math.max(1, Number(link.confirmations || 1));
        const radius = 0.014 + confidence * 0.045 + Math.min(0.028, Math.log2(confirmations + 1) * 0.007);
        const curve = curveFor(source, target, link);
        const geometry = new THREE.TubeGeometry(curve, 8, radius, 3, false);
        const material = new THREE.MeshBasicMaterial({ color: 0x58709c, transparent: true, opacity: 0.1 + confidence * 0.27, depthWrite: false });
        const edge = new THREE.Mesh(geometry, material);
        const appended = hadGraph && (!previousLinkIds.has(linkIdentity(link)) || (dreamChanged && (dreamTouched.has(link.source) || dreamTouched.has(link.target))));
        edge.userData = { link, source: link.source, target: link.target, curve, baseColor:material.color.clone(), baseOpacity:material.opacity, appendedAt:appended ? appendedAt : 0 };
        graphRoot.add(edge);
        edgeObjects.push(edge);
        edgeById.set(linkIdentity(link), edge);
        if (appended) appendPulseObjects.push(edge);
      }

      const sphereGeometry = new THREE.IcosahedronGeometry(1, 1);
      for (const node of nodes) {
        const confidence = THREE.MathUtils.clamp(Number(node.confidence ?? 0.65), 0.1, 1);
        const degree = (linksByNode.get(node.id) || []).length;
        const radius = 0.3 + Math.min(0.48, Math.log2(degree + 1) * 0.09) + confidence * 0.12;
        const material = new THREE.MeshStandardMaterial({
          color: nodeColor(node),
          emissive: nodeColor(node),
          emissiveIntensity: 0.28,
          roughness: 0.38,
          metalness: 0.08,
          transparent: true,
        });
        const mesh = new THREE.Mesh(sphereGeometry.clone(), material);
        const targetPosition = positions.get(node.id).clone();
        mesh.position.copy(previousPositions.get(node.id) || targetPosition);
        mesh.scale.setScalar(radius);
        const appended = hadGraph && (!previousNodeIds.has(node.id) || (dreamChanged && dreamTouched.has(node.id)));
        mesh.userData = { node, baseScale: radius, baseColor:material.color.clone(), displayEmissiveIntensity:0.28, appendedAt:appended ? appendedAt : 0, tweenFrom:mesh.position.clone(), tweenTarget:targetPosition, tweenAt:hadGraph && previousPositions.has(node.id) ? appendedAt : 0 };
        graphRoot.add(mesh);
        meshes.push(mesh);
        meshById.set(node.id, mesh);
        if (appended) appendPulseObjects.push(mesh);
      }
      if (hadGraph) {
        camera.position.copy(savedCamera);
        controls.target.copy(savedTarget);
        if (selectedId && meshById.has(selectedId)) {
          selected = meshById.get(selectedId);
          selected.scale.setScalar(selected.userData.baseScale * 1.7);
          makeLabel(selected.userData.node, selected);
          controls.target.copy(selected.position);
        }
        controls.update();
      } else {
        fitCamera();
      }
      applyFilter({ query: document.getElementById('graph-search')?.value || '', kind: document.getElementById('graph-kind')?.value || '' });
    }

    function fireActivations(payload) {
      const events = Array.isArray(payload?.events) ? payload.events : [];
      for (const event of events.sort((left, right) => Number(left.sequence || 0) - Number(right.sequence || 0))) {
        const sequence = Number(event.sequence || 0);
        if (!sequence || sequence <= lastActivationSequence) continue;
        lastActivationSequence = sequence;
        const now = performance.now(), intensity = THREE.MathUtils.clamp(Number(event.intensity || 1), .1, 1);
        const origins = (event.origin_node_ids || []).filter(nodeId => meshById.has(nodeId));
        const explicit = (event.node_ids || []).filter(nodeId => meshById.has(nodeId));
        const seeds = origins.length ? origins : explicit;
        const schedule = new Map(seeds.map(nodeId => [nodeId, 0]));
        for (const nodeId of explicit) if (!schedule.has(nodeId)) schedule.set(nodeId, activationHopMs);
        const queue = [...schedule].map(([nodeId, delay]) => ({nodeId,delay,depth:delay ? 1 : 0}));
        let traversed = 0;
        while (queue.length && traversed < 120) {
          const current = queue.shift(); traversed += 1;
          if (!event.cascade || current.depth >= 3) continue;
          const neighbors = [...(linksByNode.get(current.nodeId) || [])]
            .sort((left, right) => Number(right.confidence || 0) - Number(left.confidence || 0)).slice(0, 10);
          for (const link of neighbors) {
            const neighborId = link.source === current.nodeId ? link.target : link.source;
            const delay = current.delay + activationHopMs;
            const edge = edgeById.get(linkIdentity(link));
            if (edge) {
              edge.userData.activationAt = now + current.delay + activationHopMs * .45;
              edge.userData.activationIntensity = intensity * Math.pow(.78, current.depth);
              edge.userData.activationColor = new THREE.Color(firingColor(event.source));
              activePulseObjects.add(edge);
            }
            if (!schedule.has(neighborId) || delay < schedule.get(neighborId)) {
              schedule.set(neighborId, delay);
              queue.push({nodeId:neighborId,delay,depth:current.depth+1});
            }
          }
        }
        for (const [nodeId, delay] of schedule) {
          const mesh = meshById.get(nodeId);
          if (!mesh) continue;
          mesh.userData.activationAt = now + delay;
          mesh.userData.activationIntensity = intensity * Math.pow(.82, Math.round(delay / activationHopMs));
          mesh.userData.activationSource = event.source;
          mesh.userData.activationColor = new THREE.Color(firingColor(event.source));
          activePulseObjects.add(mesh);
        }
      }
    }

    function fitCamera() {
      if (!meshes.length) {
        camera.position.set(0, 0, 32);
        controls.target.set(0, 0, 0);
        controls.update();
        return;
      }
      const bounds = new THREE.Box3().setFromObject(graphRoot);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const distance = Math.max(18, size.length() * 0.78);
      camera.position.copy(center).add(new THREE.Vector3(0, size.y * 0.08, distance));
      controls.target.copy(center);
      controls.update();
    }

    function applyFilter(filter = {}) {
      const query = String(filter.query || '').trim().toLowerCase();
      const kind = String(filter.kind || '');
      const visible = new Set();
      for (const mesh of meshes) {
        const node = mesh.userData.node;
        const text = `${node.label || ''} ${node.source_id || ''} ${node.subtype || ''}`.toLowerCase();
        const match = (!kind || node.kind === kind) && (!query || text.includes(query));
        if (match) visible.add(node.id);
        mesh.material.opacity = match ? 1 : 0.075;
        mesh.userData.displayEmissiveIntensity = match ? 0.28 : 0.04;
        mesh.material.emissiveIntensity = mesh.userData.displayEmissiveIntensity;
      }
      for (const edge of edgeObjects) {
        const match = visible.has(edge.userData.source) && visible.has(edge.userData.target);
        const confidence = THREE.MathUtils.clamp(Number(edge.userData.link.confidence ?? 0.5), 0.05, 1);
        edge.userData.displayOpacity = match ? 0.1 + confidence * 0.27 : 0.012;
        edge.material.opacity = edge.userData.displayOpacity;
      }
      if ((query || kind) && visible.size === 1) {
        const target = meshById.get([...visible][0]);
        controls.target.copy(target.position);
      }
    }

    function makeLabel(node, mesh) {
      if (labelSprite) {
        labelSprite.material.map?.dispose();
        labelSprite.material.dispose();
        graphRoot.remove(labelSprite);
      }
      const text = String(node.label || node.source_id || node.id).slice(0, 64);
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      const scale = 2;
      context.font = `600 ${12 * scale}px ui-sans-serif, system-ui, sans-serif`;
      const width = Math.min(520, Math.ceil(context.measureText(text).width + 30 * scale));
      canvas.width = Math.max(160, width);
      canvas.height = 34 * scale;
      context.fillStyle = 'rgba(9,13,18,.96)';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = 'rgba(132,173,255,.75)';
      context.lineWidth = 2;
      context.stroke();
      context.fillStyle = '#f2f4f7';
      context.font = `600 ${12 * scale}px ui-sans-serif, system-ui, sans-serif`;
      context.textBaseline = 'middle';
      context.fillText(text, 15 * scale, canvas.height / 2);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
      labelSprite = new THREE.Sprite(material);
      labelSprite.scale.set(canvas.width / 75, canvas.height / 75, 1);
      labelSprite.position.copy(mesh.position).add(new THREE.Vector3(0, mesh.scale.x + 0.8, 0));
      labelSprite.renderOrder = 20;
      graphRoot.add(labelSprite);
    }

    function selectMesh(mesh) {
      if (selected && selected !== mesh) selected.scale.setScalar(selected.userData.baseScale);
      selected = mesh;
      if (!mesh) return;
      mesh.scale.setScalar(mesh.userData.baseScale * 1.7);
      makeLabel(mesh.userData.node, mesh);
      controls.target.copy(mesh.position);
      controls.update();
      const neighbors = (linksByNode.get(mesh.userData.node.id) || []).map(link => {
        const otherId = link.source === mesh.userData.node.id ? link.target : link.source;
        return { ...link, label: nodeById.get(otherId)?.label || otherId };
      });
      window.dispatchEvent(new CustomEvent('egg:graph-selection', { detail: { node: mesh.userData.node, neighbors } }));
    }

    function updatePointer(event) {
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    }

    renderer.domElement.addEventListener('pointerdown', event => { pointerDown = { x: event.clientX, y: event.clientY }; });
    renderer.domElement.addEventListener('pointermove', event => {
      updatePointer(event);
      raycaster.setFromCamera(pointer, camera);
      const next = raycaster.intersectObjects(meshes, false)[0]?.object || null;
      if (next !== hovered) {
        hovered = next;
        renderer.domElement.style.cursor = hovered ? 'pointer' : 'grab';
      }
    });
    renderer.domElement.addEventListener('pointerup', event => {
      const origin = pointerDown;
      pointerDown = null;
      if (!origin || Math.hypot(event.clientX - origin.x, event.clientY - origin.y) > 5) return;
      updatePointer(event);
      raycaster.setFromCamera(pointer, camera);
      selectMesh(raycaster.intersectObjects(meshes, false)[0]?.object || null);
    });

    function resize() {
      const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(container);
    resize();
    fitCamera();

    window.addEventListener('egg:graph-data', event => buildGraph(event.detail));
    window.addEventListener('egg:graph-activations', event => fireActivations(event.detail));
    window.addEventListener('egg:graph-filter', event => applyFilter(event.detail));
    window.addEventListener('egg:graph-reset', fitCamera);
    if (window.__eggGraphData) buildGraph(window.__eggGraphData);
    if (window.__eggGraphActivations) fireActivations(window.__eggGraphActivations);

    const clock = new THREE.Clock();
    function animate() {
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      const now = performance.now();
      let selectedMoved = false;
      for (const mesh of meshes) {
        if (!mesh.userData.tweenAt) continue;
        const raw = Math.min(1, (now - mesh.userData.tweenAt) / layoutTweenMs);
        const eased = 1 - Math.pow(1 - raw, 3);
        mesh.position.lerpVectors(mesh.userData.tweenFrom, mesh.userData.tweenTarget, eased);
        if (mesh === selected) selectedMoved = true;
        if (raw >= 1) mesh.userData.tweenAt = 0;
      }
      if (selectedMoved) {
        controls.target.lerp(selected.position, 0.22);
        if (labelSprite) labelSprite.position.copy(selected.position).add(new THREE.Vector3(0, selected.scale.x + 0.8, 0));
      }
      controls.update();
      appendPulseObjects = appendPulseObjects.filter(object => {
        const progress = (now - object.userData.appendedAt) / appendFlashMs;
        if (progress >= 1) {
          object.material.color.copy(object.userData.baseColor);
          if (object.userData.node) {
            object.material.emissive.copy(object.userData.baseColor);
            object.material.emissiveIntensity = object.userData.displayEmissiveIntensity;
            if (object !== selected) object.scale.setScalar(object.userData.baseScale);
          } else object.material.opacity = object.userData.displayOpacity ?? object.userData.baseOpacity;
          return false;
        }
        const flash = Math.max(0, 1 - progress);
        object.material.color.copy(object.userData.baseColor).lerp(white, flash);
        if (object.userData.node) {
          object.material.emissive.copy(object.userData.baseColor).lerp(white, flash);
          object.material.emissiveIntensity = 0.28 + flash * 2.8;
          if (object !== selected) object.scale.setScalar(object.userData.baseScale * (1 + flash * 0.28));
        } else {
          object.material.opacity = Math.max(object.userData.baseOpacity, 0.25 + flash * 0.75);
        }
        return true;
      });
      for (const object of [...activePulseObjects]) {
        const elapsed = now - object.userData.activationAt;
        if (elapsed < 0) continue;
        const progress = elapsed / activationPulseMs;
        if (progress >= 1) {
          object.material.color.copy(object.userData.baseColor);
          if (object.userData.node) {
            object.material.emissive.copy(object.userData.baseColor);
            object.material.emissiveIntensity = object.userData.displayEmissiveIntensity;
            if (object !== selected) object.scale.setScalar(object.userData.baseScale);
          } else object.material.opacity = object.userData.displayOpacity ?? object.userData.baseOpacity;
          activePulseObjects.delete(object);
          continue;
        }
        const firing = Math.sin(progress * Math.PI) * Number(object.userData.activationIntensity || 1);
        const pulseColor = object.userData.activationColor || white;
        object.material.color.copy(object.userData.baseColor).lerp(pulseColor, firing);
        if (object.userData.node) {
          object.material.emissive.copy(object.userData.baseColor).lerp(pulseColor, firing);
          object.material.emissiveIntensity = object.userData.displayEmissiveIntensity + firing * 3.4;
          if (object !== selected) object.scale.setScalar(object.userData.baseScale * (1 + firing * .42));
        } else object.material.opacity = Math.max(object.userData.baseOpacity, .28 + firing * .72);
      }
      if (selected) selected.scale.setScalar(selected.userData.baseScale * (1.62 + Math.sin(elapsed * 2.2) * 0.08));
      renderer.render(scene, camera);
    }
    animate();
  }
}
