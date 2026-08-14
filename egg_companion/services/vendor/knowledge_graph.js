import * as THREE from 'three';
import { OrbitControls } from '/assets/OrbitControls.js?v=20260811a';

const ASSOCIATIVE_RELATIONS = {
  has_alias: 1,
  has_label: 0.94,
  'object-label-evidence': 0.91,
  sighting: 0.88,
  observation: 0.84,
  used_for: 0.8,
  evokes_reflection: 0.74,
  participant: 0.7,
  heard_with: 0.67,
  co_observed_with: 0.62,
  appears_in_day: 0.91,
  observed_in_day: 0.86,
  read_in_day: 0.88,
  heard_in_day: 0.88,
  experienced_day: 0.94,
  replays_day: 0.96,
  contributes_to_story: 0.93,
  precedes_day: 0.82,
};
const RELATION_GEOMETRY = {
  identity: { angle:0.04, arch:0.14 },
  observation: { angle:0.58, arch:0.34 },
  co_presence: { angle:-0.72, arch:0.64 },
  audio: { angle:-1.18, arch:0.48 },
  temporal: { angle:1.2, arch:0.56 },
  reflective: { angle:2.18, arch:0.42 },
  associative: { angle:0.3, arch:0.3 },
};
let associativeLayoutCache = { signature:'', result:null };

function stableHash(value) {
  let result = 2166136261;
  for (const character of String(value)) {
    result ^= character.charCodeAt(0);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

function seededUnit(seed) {
  let value = seed || 1;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function unitVector(seed) {
  const random = seededUnit(seed);
  const z = random() * 2 - 1;
  const angle = random() * Math.PI * 2;
  const radius = Math.sqrt(Math.max(0, 1 - z * z));
  return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, z };
}

function relationFamily(relation) {
  if (/alias|label|identity|named|same_as/.test(relation)) return 'identity';
  if (/heard|audio|speech|voice|sound/.test(relation)) return 'audio';
  if (/co[_-]?(observed|present|occur)|shared_context/.test(relation)) return 'co_presence';
  if (/episode|participant|temporal|before|after|during|day|dream|replay|story|precedes/.test(relation)) return 'temporal';
  if (/reflection|claim|used_for|caused|supports|contradicts/.test(relation)) return 'reflective';
  if (/sighting|observation|evidence|appearance|ocr|content|detected/.test(relation)) return 'observation';
  return 'associative';
}

function relationRule(link) {
  const relation = String(link.relation || '').toLowerCase();
  const semanticAffinity = ASSOCIATIVE_RELATIONS[relation] || 0.58;
  const confidence = Math.max(0.04, Math.min(1, Number(link.confidence ?? 0.5)));
  const confirmations = Math.max(1, Number(link.confirmations ?? link.recurrence ?? link.count ?? 1));
  const recurrence = Math.min(1, Math.log2(confirmations + 1) / 10);
  const tightness = Math.max(0.08, Math.min(1, semanticAffinity * 0.34 + confidence * 0.5 + recurrence * 0.16));
  const associationStrength = Math.max(0.05, Math.min(1, tightness * 0.5 + confidence * 0.22 + recurrence * 0.28));
  const thickness = Math.max(0.06, Math.min(1, associationStrength * 0.62 + recurrence * 0.38));
  const family = relationFamily(relation), geometry = RELATION_GEOMETRY[family];
  const angleJitter = ((stableHash(`${relation}:${link.source}:${link.target}`) % 1001) / 1000 - 0.5) * 0.16;
  const arch = Math.max(0.08, Math.min(1, geometry.arch * 0.32 + recurrence * 0.5 + (1 - tightness) * 0.18));
  return {
    tightness,
    distance: 2.35 + Math.pow(1 - tightness, 1.3) * 9.2,
    strength: 0.25 + tightness * 1.55,
    associationStrength,
    confidence,
    confirmations,
    recurrence,
    thickness,
    arch,
    angle: geometry.angle + angleJitter,
    family,
  };
}

/*
 * Spline channels stay independent: the arch rises with recurrence, its plane
 * rotates by relationship family, and its rendered radius follows evidence
 * strength. Keeping this in plain objects makes WebGL and canvas agree.
 */
function associativeSplinePoints(source, target, link, associative = relationRule(link)) {
  const dx = target.x - source.x, dy = target.y - source.y, dz = (target.z || 0) - (source.z || 0);
  const chordLength = Math.max(0.001, Math.hypot(dx, dy, dz));
  const axis = { x:dx / chordLength, y:dy / chordLength, z:dz / chordLength };
  const reference = Math.abs(axis.y) < 0.92 ? { x:0, y:1, z:0 } : { x:1, y:0, z:0 };
  const projection = reference.x * axis.x + reference.y * axis.y + reference.z * axis.z;
  let normalA = { x:reference.x - axis.x * projection, y:reference.y - axis.y * projection, z:reference.z - axis.z * projection };
  const normalLength = Math.max(0.001, Math.hypot(normalA.x, normalA.y, normalA.z));
  normalA = { x:normalA.x / normalLength, y:normalA.y / normalLength, z:normalA.z / normalLength };
  const normalB = {
    x:axis.y * normalA.z - axis.z * normalA.y,
    y:axis.z * normalA.x - axis.x * normalA.z,
    z:axis.x * normalA.y - axis.y * normalA.x,
  };
  const angle = Number(associative.angle || 0), cosine = Math.cos(angle), sine = Math.sin(angle);
  const archDirection = { x:normalA.x * cosine + normalB.x * sine, y:normalA.y * cosine + normalB.y * sine, z:normalA.z * cosine + normalB.z * sine };
  const archHeight = Math.max(0.22, Math.min(7, chordLength * (0.035 + Number(associative.arch || 0.2) * 0.2)));
  const pointAt = (fraction, lift) => ({
    x:source.x + dx * fraction + archDirection.x * archHeight * lift,
    y:source.y + dy * fraction + archDirection.y * archHeight * lift,
    z:(source.z || 0) + dz * fraction + archDirection.z * archHeight * lift,
  });
  return [
    { x:source.x, y:source.y, z:source.z || 0 },
    pointAt(0.26, 0.72),
    pointAt(0.5, 1),
    pointAt(0.74, 0.72),
    { x:target.x, y:target.y, z:target.z || 0 },
  ];
}

function associativeLinkKey(link) {
  return String(link.id || `${link.source}:${link.relation || ''}:${link.target}`);
}

/*
 * Deterministic 3-D associative layout shared by WebGL and the canvas fallback.
 * Connected memory neighborhoods occupy separate volumes; edge semantics decide
 * proximity inside each volume. Time is a gentle component-local direction, not
 * a global depth plane, and high-degree concepts settle nearer a neighborhood's
 * core while evidence and episodes remain readable satellites.
 */
function associativeLayout3D(rawNodes, rawLinks, layoutRevision = '') {
  const orderedNodes = [...rawNodes].sort((left, right) => String(left.id).localeCompare(String(right.id)));
  const nodeIds = new Set(orderedNodes.map(node => node.id));
  const usableLinks = rawLinks
    .filter(link => nodeIds.has(link.source) && nodeIds.has(link.target) && link.source !== link.target)
    .map(link => ({ ...link, layout: relationRule(link) }))
    .sort((left, right) => String(left.id || `${left.source}:${left.target}`).localeCompare(String(right.id || `${right.source}:${right.target}`)));
  let signatureHash = 2166136261;
  const includeSignature = value => {
    for (const character of String(value)) { signatureHash ^= character.charCodeAt(0); signatureHash = Math.imul(signatureHash, 16777619); }
  };
  for (const node of orderedNodes) includeSignature(`${node.id}:${node.kind || ''}:${node.subtype || ''};`);
  for (const link of usableLinks) {
    const confidenceBand = Math.round(link.layout.confidence * 20);
    const recurrenceBand = Math.round(link.layout.recurrence * 20);
    includeSignature(`${link.id || ''}:${link.source}:${link.relation || ''}:${link.target}:${confidenceBand}:${recurrenceBand}:${link.layout.family};`);
  }
  const signature = `${orderedNodes.length}:${usableLinks.length}:${signatureHash >>> 0}:${stableHash(layoutRevision)}`;
  if (associativeLayoutCache.signature === signature && associativeLayoutCache.result) return associativeLayoutCache.result;
  const neighborIds = new Map(orderedNodes.map(node => [node.id, new Set()]));
  for (const link of usableLinks) {
    neighborIds.get(link.source).add(link.target);
    neighborIds.get(link.target).add(link.source);
  }
  for (const link of usableLinks) {
    const sourceNeighbors = neighborIds.get(link.source), targetNeighbors = neighborIds.get(link.target);
    const smaller = sourceNeighbors.size < targetNeighbors.size ? sourceNeighbors : targetNeighbors;
    const larger = smaller === sourceNeighbors ? targetNeighbors : sourceNeighbors;
    let shared = 0;
    for (const id of smaller) if (larger.has(id)) shared += 1;
    const structuralAgreement = shared / Math.max(1, Math.min(sourceNeighbors.size, targetNeighbors.size));
    link.layout.tightness = Math.min(1, link.layout.tightness * 0.84 + structuralAgreement * 0.16);
    link.layout.distance = 2.35 + Math.pow(1 - link.layout.tightness, 1.3) * 9.2;
    link.layout.strength = 0.25 + link.layout.tightness * 1.55;
  }
  const adjacency = new Map(orderedNodes.map(node => [node.id, []]));
  const weightedDegree = new Map(orderedNodes.map(node => [node.id, 0]));
  for (const link of usableLinks) {
    adjacency.get(link.source).push({ id:link.target, weight:link.layout.strength, tightness:link.layout.tightness, link });
    adjacency.get(link.target).push({ id:link.source, weight:link.layout.strength, tightness:link.layout.tightness, link });
    weightedDegree.set(link.source, weightedDegree.get(link.source) + link.layout.strength);
    weightedDegree.set(link.target, weightedDegree.get(link.target) + link.layout.strength);
  }

  const components = [];
  const visited = new Set();
  for (const node of orderedNodes) {
    if (visited.has(node.id)) continue;
    const ids = [], queue = [node.id];
    visited.add(node.id);
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const id = queue[cursor];
      ids.push(id);
      for (const neighbor of adjacency.get(id)) if (!visited.has(neighbor.id)) { visited.add(neighbor.id); queue.push(neighbor.id); }
    }
    ids.sort();
    components.push({ ids, key: ids[0], weight: ids.reduce((sum, id) => sum + weightedDegree.get(id), 0) });
  }
  components.sort((left, right) => right.weight - left.weight || right.ids.length - left.ids.length || left.key.localeCompare(right.key));

  const componentByNode = new Map(), componentOrigins = new Map(), temporalAxes = new Map();
  const volumePoint = (key, span) => {
    const random = seededUnit(stableHash(key));
    const a = random() * 2 - 1, b = random() * 2 - 1, c = random() * 2 - 1;
    return {
      x: (a + Math.sin((b + c) * Math.PI) * 0.16) * span,
      y: (b + Math.sin((c + a) * Math.PI) * 0.16) * span,
      z: (c + Math.sin((a + b) * Math.PI) * 0.16) * span,
    };
  };
  components.forEach((component, index) => {
    component.ids.forEach(id => componentByNode.set(id, component));
    componentOrigins.set(component.key, index === 0 && component.ids.length > 1 ? { x:0, y:0, z:0 } : volumePoint(`component:${component.key}`, component.ids.length > 1 ? 34 : 50));
    temporalAxes.set(component.key, unitVector(stableHash(`time:${component.key}`)));
  });

  // A few deterministic weighted-label passes identify local associative
  // neighborhoods without imposing categories or geometric lobes.
  let communityLabels = new Map();
  for (const node of orderedNodes) {
    const neighbors = adjacency.get(node.id);
    if (!neighbors.length) { communityLabels.set(node.id, node.id); continue; }
    const strongest = [...neighbors].sort((left, right) =>
      right.tightness * (1 + Math.log2(weightedDegree.get(right.id) + 1)) - left.tightness * (1 + Math.log2(weightedDegree.get(left.id) + 1)) || left.id.localeCompare(right.id)
    )[0];
    communityLabels.set(node.id, strongest.tightness >= 0.52 ? strongest.id : node.id);
  }
  for (let pass = 0; pass < 3; pass += 1) {
    const nextLabels = new Map(communityLabels);
    for (const node of orderedNodes) {
      const neighbors = adjacency.get(node.id);
      if (!neighbors.length) continue;
      const scores = new Map([[communityLabels.get(node.id), weightedDegree.get(node.id) * 0.18]]);
      for (const neighbor of neighbors) {
        const label = communityLabels.get(neighbor.id);
        scores.set(label, (scores.get(label) || 0) + neighbor.tightness * neighbor.tightness);
      }
      const winner = [...scores].sort((left, right) => right[1] - left[1] || String(left[0]).localeCompare(String(right[0])))[0];
      nextLabels.set(node.id, winner[0]);
    }
    communityLabels = nextLabels;
  }
  const communityAffinity = new Map();
  for (const node of orderedNodes) {
    const neighbors = adjacency.get(node.id), ownLabel = communityLabels.get(node.id);
    const total = neighbors.reduce((sum, neighbor) => sum + neighbor.tightness, 0);
    const internal = neighbors.reduce((sum, neighbor) => sum + (communityLabels.get(neighbor.id) === ownLabel ? neighbor.tightness : 0), 0);
    communityAffinity.set(node.id, total ? internal / total : 0);
  }

  const nodeById = new Map(orderedNodes.map(node => [node.id, node]));
  const timestamps = orderedNodes.map(node => Date.parse(node.updated_at || '')).filter(Number.isFinite);
  const oldest = timestamps.length ? Math.min(...timestamps) : 0, newest = timestamps.length ? Math.max(...timestamps) : 0;
  const positions = new Map(), masses = new Map();
  for (const node of orderedNodes) masses.set(node.id, 1 + Math.sqrt(weightedDegree.get(node.id)));
  for (const component of components) {
    const origin = componentOrigins.get(component.key), temporalAxis = temporalAxes.get(component.key);
    if (component.ids.length === 1) {
      positions.set(component.ids[0], volumePoint(`unassociated:${component.ids[0]}`, 52));
      continue;
    }
    const root = [...component.ids].sort((left, right) => weightedDegree.get(right) - weightedDegree.get(left) || left.localeCompare(right))[0];
    positions.set(root, { ...origin });
    const discovered = new Set([root]), queue = [root];
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const sourceId = queue[cursor], source = positions.get(sourceId);
      const neighbors = [...adjacency.get(sourceId)].sort((left, right) => right.tightness - left.tightness || left.id.localeCompare(right.id));
      for (const neighbor of neighbors) {
        if (discovered.has(neighbor.id)) continue;
        discovered.add(neighbor.id); queue.push(neighbor.id);
        const branch = unitVector(stableHash(`branch:${sourceId}:${neighbor.id}:${associativeLinkKey(neighbor.link)}`));
        const random = seededUnit(stableHash(`length:${sourceId}:${neighbor.id}`));
        const length = neighbor.link.layout.distance * (0.88 + random() * 0.24);
        const timestamp = Date.parse(nodeById.get(neighbor.id)?.updated_at || '');
        const temporal = newest > oldest && Number.isFinite(timestamp) ? ((timestamp - oldest) / (newest - oldest) - 0.5) * 1.8 : 0;
        positions.set(neighbor.id, {
          x: source.x + branch.x * length + temporalAxis.x * temporal,
          y: source.y + branch.y * length + temporalAxis.y * temporal,
          z: source.z + branch.z * length + temporalAxis.z * temporal,
        });
      }
    }
  }

  const forceNodes = orderedNodes.filter(node => weightedDegree.get(node.id) > 0);
  const forceCount = forceNodes.length;
  const iterationCount = forceCount > 1200 ? 28 : forceCount > 500 ? 36 : 52;
  const repulsionSamples = Math.max(0, Math.min(forceCount - 1, forceCount > 1200 ? 4 : forceCount > 500 ? 6 : 10));
  const repulsionSeeds = forceNodes.map(node => Array.from({ length:repulsionSamples }, (_, sample) => stableHash(`${node.id}:${sample}`) + sample * 97));
  for (let iteration = 0; iteration < iterationCount; iteration += 1) {
    const temperature = 0.24 * Math.pow(1 - iteration / (iterationCount + 8), 1.35);
    for (const link of usableLinks) {
      const source = positions.get(link.source), target = positions.get(link.target);
      let dx = target.x - source.x, dy = target.y - source.y, dz = target.z - source.z;
      let distance = Math.hypot(dx, dy, dz);
      if (distance < 0.001) {
        const nudge = unitVector(stableHash(`link:${link.id || `${link.source}:${link.target}`}`));
        dx = nudge.x * 0.01; dy = nudge.y * 0.01; dz = nudge.z * 0.01; distance = 0.01;
      }
      const adjustment = Math.max(-2.5, Math.min(2.5, (distance - link.layout.distance) * temperature * (0.1 + link.layout.strength * 0.16)));
      const sourceMass = masses.get(link.source), targetMass = masses.get(link.target), totalMass = sourceMass + targetMass;
      const sx = dx / distance * adjustment, sy = dy / distance * adjustment, sz = dz / distance * adjustment;
      source.x += sx * targetMass / totalMass; source.y += sy * targetMass / totalMass; source.z += sz * targetMass / totalMass;
      target.x -= sx * sourceMass / totalMass; target.y -= sy * sourceMass / totalMass; target.z -= sz * sourceMass / totalMass;
    }
    const communityCenters = new Map();
    for (const node of forceNodes) {
      const label = communityLabels.get(node.id), position = positions.get(node.id), center = communityCenters.get(label) || { x:0, y:0, z:0, weight:0 };
      const weight = Math.max(0.2, communityAffinity.get(node.id));
      center.x += position.x * weight; center.y += position.y * weight; center.z += position.z * weight; center.weight += weight;
      communityCenters.set(label, center);
    }
    for (let index = 0; index < forceCount; index += 1) {
      const node = forceNodes[index], position = positions.get(node.id), mass = masses.get(node.id);
      for (let sample = 0; sample < repulsionSamples; sample += 1) {
        const offset = 1 + ((repulsionSeeds[index][sample] + iteration * 37) % Math.max(1, forceCount - 1));
        const otherNode = forceNodes[(index + offset) % forceCount], other = positions.get(otherNode.id);
        let dx = position.x - other.x, dy = position.y - other.y, dz = position.z - other.z;
        let distance = Math.hypot(dx, dy, dz);
        if (distance < 0.001) {
          const nudge = unitVector(stableHash(`apart:${node.id}:${otherNode.id}`));
          dx = nudge.x * 0.01; dy = nudge.y * 0.01; dz = nudge.z * 0.01; distance = 0.01;
        }
        const clearance = 2.15 + Math.min(1.4, Math.log2(mass + masses.get(otherNode.id)) * 0.28);
        if (distance >= clearance * 2.35) continue;
        const push = Math.min(0.7, temperature * (0.34 / Math.max(0.18, distance) + Math.max(0, clearance - distance) * 0.24));
        position.x += dx / distance * push; position.y += dy / distance * push; position.z += dz / distance * push;
        other.x -= dx / distance * push; other.y -= dy / distance * push; other.z -= dz / distance * push;
      }
      const center = communityCenters.get(communityLabels.get(node.id)), affinity = communityAffinity.get(node.id);
      if (center?.weight && affinity > 0) {
        const cohesion = temperature * 0.026 * affinity;
        position.x += (center.x / center.weight - position.x) * cohesion;
        position.y += (center.y / center.weight - position.y) * cohesion;
        position.z += (center.z / center.weight - position.z) * cohesion;
      }
      const componentOrigin = componentOrigins.get(componentByNode.get(node.id).key);
      const gravity = temperature * 0.0015;
      position.x += (componentOrigin.x - position.x) * gravity;
      position.y += (componentOrigin.y - position.y) * gravity;
      position.z += (componentOrigin.z - position.z) * gravity;
    }
  }

  // Keep the aggregate cloud volumetric even when one dense relationship family
  // would otherwise collapse depth. The cap avoids turning sparse chains into walls.
  if (positions.size > 3) {
    const values = [...positions.values()];
    const mean = values.reduce((sum, position) => ({ x: sum.x + position.x, y: sum.y + position.y, z: sum.z + position.z }), { x: 0, y: 0, z: 0 });
    mean.x /= values.length; mean.y /= values.length; mean.z /= values.length;
    const variance = values.reduce((sum, position) => ({ x: sum.x + (position.x - mean.x) ** 2, y: sum.y + (position.y - mean.y) ** 2, z: sum.z + (position.z - mean.z) ** 2 }), { x: 0, y: 0, z: 0 });
    const maximum = Math.max(variance.x, variance.y, variance.z, 0.001);
    const gains = {
      x: Math.min(2.25, Math.sqrt(maximum * 0.24 / Math.max(variance.x, 0.001))),
      y: Math.min(2.25, Math.sqrt(maximum * 0.24 / Math.max(variance.y, 0.001))),
      z: Math.min(2.25, Math.sqrt(maximum * 0.24 / Math.max(variance.z, 0.001))),
    };
    for (const position of values) {
      position.x = mean.x + (position.x - mean.x) * Math.max(1, gains.x);
      position.y = mean.y + (position.y - mean.y) * Math.max(1, gains.y);
      position.z = mean.z + (position.z - mean.z) * Math.max(1, gains.z);
    }
  }
  const result = { positions, degree:weightedDegree, links:new Map(usableLinks.map(link => [associativeLinkKey(link), link.layout])) };
  associativeLayoutCache = { signature, result };
  return result;
}

function initCanvasFallback(container) {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', 'Interactive compatibility rendering of the multimodal knowledge graph');
  container.replaceChildren(canvas);
  const context = canvas.getContext('2d', { alpha: false });
  const colors = { person:'#60a5fa', appearance:'#38bdf8', object:'#34d399', object_category:'#2dd4bf', sound_event:'#ffae00', content:'#ffae00', daily_narrative:'#f97316', dream_replay:'#8b5cf6', cognitive:'#06b6d4', evidence:'#c084fc', claim:'#fb7185', episode:'#94a3b8', entity:'#22d3ee' };
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
    if (subtype.includes('daily_narrative')) return colors.daily_narrative;
    if (subtype.includes('dream_replay')) return colors.dream_replay;
    if (subtype.includes('cognitive_document') || subtype.includes('abstraction') || subtype.includes('reflection')) return colors.cognitive;
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
    const association = associativeLayout3D(data.nodes || [], data.links || [], dreamRevision);
    nodes = (data.nodes || []).map((node, index) => {
      const position = association.positions.get(node.id) || { x:0, y:0, z:0 };
      return { ...node, ...position, degree:association.degree.get(node.id) || 0, index, appendedAt:hadGraph && (!previousNodeIds.has(node.id) || (dreamChanged && dreamTouched.has(node.id))) ? appendedAt : 0 };
    });
    const byId = new Map(nodes.map(node => [node.id, node]));
    links = (data.links || []).filter(link => byId.has(link.source) && byId.has(link.target)).map(link => ({ ...link, associative:association.links.get(associativeLinkKey(link)) || relationRule(link), sourceNode:byId.get(link.source), targetNode:byId.get(link.target), appendedAt:hadGraph && !previousLinkIds.has(linkIdentity(link)) ? appendedAt : 0 }));
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
          .sort((left, right) => Number(right.link.associative?.tightness || 0) - Number(left.link.associative?.tightness || 0)).slice(0, 10);
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
      const associative = link.associative || relationRule(link);
      const spline = associativeSplinePoints(link.sourceNode, link.targetNode, link, associative).map(point);
      const source = spline[0], controlA = spline[1], controlB = spline[3], target = spline[4];
      const visible = matches(link.sourceNode) && matches(link.targetNode);
      context.beginPath(); context.moveTo(source.x, source.y); context.bezierCurveTo(controlA.x, controlA.y, controlB.x, controlB.y, target.x, target.y);
      context.strokeStyle = visible ? `rgba(102,126,168,${.08 + associative.associationStrength * .34 + associative.recurrence * .12})` : 'rgba(68,82,108,.025)';
      context.lineWidth = visible ? .45 + associative.thickness * 4.4 : .35;
      context.stroke();
      const flash = appendFlash(link);
      if (flash > 0 && visible) { context.strokeStyle = `rgba(255,255,255,${flash})`; context.lineWidth += 1.8 * flash; context.stroke(); }
      const firing = activationPulse(link, now);
      if (firing > 0 && visible) {
        context.globalAlpha = Math.min(1,.22+firing); context.strokeStyle = link.activationColor || '#fff'; context.lineWidth += 2.6 * firing; context.stroke();context.globalAlpha=1;
        const raw = Math.max(0, Math.min(1, (now-link.activationAt)/activationPulseMs));
        const t = link.activationFrom === link.target ? 1-raw : raw, inverse = 1-t;
        const sparkX=inverse**3*source.x+3*inverse*inverse*t*controlA.x+3*inverse*t*t*controlB.x+t**3*target.x;
        const sparkY=inverse**3*source.y+3*inverse*inverse*t*controlA.y+3*inverse*t*t*controlB.y+t**3*target.y;
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
      daily_narrative: 0xf97316,
      dream_replay: 0x8b5cf6,
      cognitive: 0x06b6d4,
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
      if (subtype.includes('daily_narrative')) return colors.daily_narrative;
      if (subtype.includes('dream_replay')) return colors.dream_replay;
      if (subtype.includes('cognitive_document') || subtype.includes('abstraction') || subtype.includes('reflection')) return colors.cognitive;
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

    function relationshipLayout(nodes, links, revision = '') {
      const association = associativeLayout3D(nodes, links, revision);
      return {
        positions:new Map([...association.positions].map(([id, position]) => [id, new THREE.Vector3(position.x, position.y, position.z)])),
        links:association.links,
      };
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

    function curveFor(source, target, link, associative) {
      const points = associativeSplinePoints(source, target, link, associative).map(point => new THREE.Vector3(point.x, point.y, point.z));
      return new THREE.CatmullRomCurve3(points, false, 'centripetal', 0.5);
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
      const links = Array.isArray(graphData.links) ? graphData.links.map(link => ({ ...link })) : [];
      const dreamTouched = new Set(graphData.dream?.touched_node_ids || []);
      if (dreamChanged) for (const link of links) if (dreamTouched.has(link.source) || dreamTouched.has(link.target)) { dreamTouched.add(link.source); dreamTouched.add(link.target); }
      const association = relationshipLayout(nodes, links, dreamRevision), positions = association.positions;
      for (const link of links) link.associative = association.links.get(associativeLinkKey(link)) || relationRule(link);
      nodes.forEach(node => nodeById.set(node.id, node));
      for (const link of links) {
        if (!linksByNode.has(link.source)) linksByNode.set(link.source, []);
        if (!linksByNode.has(link.target)) linksByNode.set(link.target, []);
        linksByNode.get(link.source).push(link);
        linksByNode.get(link.target).push(link);
      }

      const renderedLinks = links
        .filter(link => positions.has(link.source) && positions.has(link.target))
        .sort((a, b) => Number(b.associative?.tightness || 0) - Number(a.associative?.tightness || 0))
        .slice(0, 1800);
      for (const link of renderedLinks) {
        const source = positions.get(link.source), target = positions.get(link.target);
        const associative = link.associative || relationRule(link);
        const tightness = associative.tightness;
        const radius = 0.009 + associative.thickness * 0.105;
        const curve = curveFor(source, target, link, link.associative);
        const geometry = new THREE.TubeGeometry(curve, 12, radius, 4, false);
        const baseOpacity = 0.055 + associative.associationStrength * 0.27 + associative.recurrence * 0.12;
        const material = new THREE.MeshBasicMaterial({ color: 0x58709c, transparent: true, opacity: baseOpacity, depthWrite: false });
        const edge = new THREE.Mesh(geometry, material);
        const appended = hadGraph && (!previousLinkIds.has(linkIdentity(link)) || (dreamChanged && (dreamTouched.has(link.source) || dreamTouched.has(link.target))));
        edge.userData = { link, tightness, associationStrength:associative.associationStrength, recurrence:associative.recurrence, confirmations:associative.confirmations, thickness:associative.thickness, arch:associative.arch, angle:associative.angle, family:associative.family, source: link.source, target: link.target, curve, baseColor:material.color.clone(), baseOpacity, appendedAt:appended ? appendedAt : 0 };
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
            .sort((left, right) => Number(right.associative?.tightness || 0) - Number(left.associative?.tightness || 0)).slice(0, 10);
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
        edge.userData.displayOpacity = match ? edge.userData.baseOpacity : 0.012;
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
