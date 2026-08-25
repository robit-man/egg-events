from __future__ import annotations


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#07090c">
  <title>Control Center</title>
  <style>
    :root {
      color-scheme: dark;
      --canvas: #07090c;
      --surface: #0d1117;
      --surface-muted: #090d12;
      --sidebar: #07090c;
      --sidebar-soft: #121820;
      --text: #e5e7eb;
      --muted: #8b949e;
      --subtle: #596270;
      --line: #252c34;
      --line-strong: #39424d;
      --accent: #ffae00;
      --accent-soft: #2d2205;
      --success: #65d080;
      --success-soft: #102418;
      --warning: #ffae00;
      --warning-soft: #2d2205;
      --danger: #ff6b6b;
      --danger-soft: #2b1215;
      --radius: 0;
      --shadow: none;
    }

    * { box-sizing: border-box; border-radius: 0 !important; }
    html { min-width: 320px; background: var(--canvas); }
    body {
      margin: 0;
      color: var(--text);
      background: var(--canvas);
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      -webkit-font-smoothing: antialiased;
    }
    button, input, select { font: inherit; }
    button { color: inherit; }
    a { color: inherit; text-decoration: none; }
    [hidden] { display: none !important; }

    .app { min-height: 100vh; }
    .sidebar {
      position: fixed;
      inset: 0 auto 0 0;
      z-index: 40;
      display: flex;
      width: 252px;
      flex-direction: column;
      padding: 20px 14px 16px;
      color: #d0d5dd;
      background: var(--sidebar);
      border-right: 1px solid #344054;
      overflow-y: auto;
    }
    .sidebar-mark {
      display: grid;
      width: 38px;
      height: 38px;
      margin: 0 8px 24px;
      place-items: center;
      border: 1px solid #475467;
      border-radius: 12px;
      background: #182230;
    }
    .sidebar-mark::before {
      width: 14px;
      height: 19px;
      content: "";
      border: 2px solid #84adff;
      border-radius: 55% 55% 48% 48%;
      transform: translateY(1px);
    }
    .nav-label {
      padding: 0 12px 8px;
      color: #667085;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .09em;
      text-transform: uppercase;
    }
    .nav { display: grid; gap: 4px; }
    .nav-link {
      display: flex;
      min-height: 42px;
      align-items: center;
      gap: 12px;
      padding: 9px 12px;
      color: #b9c0cc;
      border-radius: 9px;
      font-weight: 550;
      transition: color .15s ease, background .15s ease;
    }
    .nav-link:hover { color: #fff; background: #1d2939; }
    .nav-link.active { color: #ffae00; background: #211b0d; box-shadow: inset 3px 0 #ffae00; }
    .nav-link svg { width: 18px; height: 18px; flex: 0 0 auto; stroke-width: 1.8; }
    .sidebar-footer { margin-top: auto; padding: 14px 10px 2px; border-top: 1px solid #344054; }
    .connection { display: flex; align-items: center; gap: 9px; color: #d0d5dd; font-size: 12px; }
    .connection-dot { width: 8px; height: 8px; border-radius: 50%; background: #f79009; box-shadow: 0 0 0 3px rgb(247 144 9 / .15); }
    .connection-dot.online { background: #32d583; box-shadow: 0 0 0 3px rgb(50 213 131 / .14); }
    .connection-dot.offline { background: #f04438; box-shadow: 0 0 0 3px rgb(240 68 56 / .14); }
    .sidebar-meta { margin: 7px 0 0 17px; color: #667085; font-size: 11px; }

    .main { min-height: 100vh; margin-left: 252px; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 30;
      display: flex;
      min-height: 72px;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 14px clamp(18px, 3vw, 40px);
      background: rgb(9 13 18 / .94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(14px);
    }
    .page-title { margin: 0; font-size: 22px; font-weight: 680; letter-spacing: -.025em; }
    .top-actions { display: flex; align-items: center; gap: 12px; }
    .sync-label { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 32px;
      padding: 6px 10px;
      color: var(--success);
      background: var(--success-soft);
      border: 1px solid #abefc6;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
    }
    .status-pill.degraded { color: var(--warning); background: var(--warning-soft); border-color: #fedf89; }
    .status-pill.offline { color: var(--danger); background: var(--danger-soft); border-color: #fecdca; }
    .mobile-menu { display: none; width: 38px; height: 38px; place-items: center; padding: 0; background: #fff; border: 1px solid var(--line); border-radius: 9px; }

    .content { width: min(1600px, 100%); margin: 0 auto; padding: 28px clamp(18px, 3vw, 40px) 48px; }
    .page { display: none; animation: page-in .18s ease; }
    .page.active { display: block; }
    @keyframes page-in { from { opacity: .35; transform: translateY(3px); } }
    .page-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 22px; }
    .page-heading h2 { margin: 0; font-size: 18px; letter-spacing: -.015em; }
    .page-heading p { max-width: 700px; margin: 5px 0 0; color: var(--muted); }

    .grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 16px; }
    .span-12 { grid-column: span 12; }
    .span-8 { grid-column: span 8; }
    .span-7 { grid-column: span 7; }
    .span-6 { grid-column: span 6; }
    .span-5 { grid-column: span 5; }
    .span-4 { grid-column: span 4; }
    .span-3 { grid-column: span 3; }
    .stack { display: grid; gap: 16px; align-content: start; }
    .card {
      min-width: 0;
      padding: 18px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; margin-bottom: 16px; }
    .card-title { margin: 0; font-size: 14px; font-weight: 680; letter-spacing: -.01em; }
    .card-note { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
    .metric-card { padding: 16px 18px; }
    .metric-label { color: var(--muted); font-size: 12px; font-weight: 600; }
    .metric-value { margin-top: 7px; font-size: clamp(23px, 3vw, 30px); font-weight: 680; letter-spacing: -.035em; }
    .metric-detail { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .metric-indicator { float: right; width: 9px; height: 9px; margin-top: 4px; border-radius: 50%; background: var(--subtle); }
    .metric-indicator.good { background: #12b76a; }
    .metric-indicator.warn { background: #f79009; }
    .metric-indicator.bad { background: #f04438; }

    .badge-row { display: flex; flex-wrap: wrap; gap: 7px; }
    .badge {
      display: inline-flex;
      min-height: 25px;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      color: #344054;
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 550;
    }
    .badge.good { color: var(--success); background: var(--success-soft); border-color: #abefc6; }
    .badge.warn { color: var(--warning); background: var(--warning-soft); border-color: #fedf89; }
    .badge.bad { color: var(--danger); background: var(--danger-soft); border-color: #fecdca; }
    .empty { padding: 30px 16px; color: var(--muted); text-align: center; background: var(--surface-muted); border: 1px dashed var(--line-strong); border-radius: 10px; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
    .pre { overflow: auto; max-height: 390px; margin: 0; padding: 14px; white-space: pre-wrap; color: #344054; background: var(--surface-muted); border: 1px solid var(--line); border-radius: 9px; }

    .camera-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .camera { overflow: hidden; background: #090d14; border: 1px solid #344054; border-radius: 11px; }
    .camera-head { display: flex; min-height: 42px; align-items: center; justify-content: space-between; gap: 10px; padding: 9px 12px; color: #eaecf0; background: #101828; }
    .camera-head .muted { color: #98a2b3; font-size: 11px; }
    .camera-stage { position: relative; overflow: hidden; background: #030712; isolation: isolate; }
    .camera-raw { display: block; width: 100%; height: 100%; object-fit: fill; }
    .camera-overlay { position: absolute; inset: 0; pointer-events: none; }
    .mask-layer { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
    .mask { fill: rgb(37 99 235 / .23); stroke: #7dd3fc; stroke-width: 2; vector-effect: non-scaling-stroke; }
    .mask-label { fill: #fff; font-size: 17px; font-weight: 700; paint-order: stroke; stroke: #101828; stroke-width: 5; stroke-linejoin: round; }
    .pose-bone { stroke: rgb(250 204 21 / .7); stroke-width: 2; vector-effect: non-scaling-stroke; stroke-linecap: round; }
    .pose-joint { fill: rgb(239 68 68 / .85); }
    .pose-joint-label { fill: #fbbf24; font-size: 11px; font-weight: 600; paint-order: stroke; stroke: #0f172a; stroke-width: 3; stroke-linejoin: round; }
    .camera-meta { min-height: 44px; flex-wrap: wrap; color: #d0d5dd; background: #101828; }

    .wave { display: block; width: 100%; height: 180px; background: #0b1220; border-radius: 10px; }
    .conversation { display: grid; gap: 12px; max-height: 620px; overflow: auto; align-content: start; }
    .message { max-width: 88%; padding: 11px 13px; border-radius: 12px; }
    .message.heard { justify-self: start; color: #344054; background: #f2f4f7; border-bottom-left-radius: 4px; }
    .message.agent { justify-self: end; color: #194185; background: var(--accent-soft); border-bottom-right-radius: 4px; }
    .message-role { display: block; margin-bottom: 4px; color: var(--muted); font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
    .message-meta { display: block; margin-top: 6px; color: var(--muted); font-size: 9px; }
    .message-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
    .message-tag { border: 1px solid #39404a; padding: 2px 5px; color: #aeb6c2; background: #0b0e12; font-size: 9px; line-height: 1.25; }
    .message-tag.tool { border-color: #74520b; color: #ffca63; }
    .message-tag.memory { border-color: #59406f; color: #d8b4fe; }
    .message-tag.association { border-color: #28556c; color: #7dd3fc; }
    .message.suppressed { opacity: .55; border: 1px dashed var(--line-strong); }

    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .field { display: grid; gap: 6px; }
    .field.full { grid-column: 1 / -1; }
    .field label { color: #475467; font-size: 12px; font-weight: 620; }
    .input, .select {
      width: 100%;
      min-width: 0;
      height: 40px;
      padding: 8px 10px;
      color: var(--text);
      background: #fff;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      outline: none;
    }
    .input:focus, .select:focus { border-color: #84adff; box-shadow: 0 0 0 3px rgb(37 99 235 / .1); }
    .button-row { display: flex; flex-wrap: wrap; align-items: center; gap: 9px; grid-column: 1 / -1; }
    .button {
      display: inline-flex;
      min-height: 38px;
      align-items: center;
      justify-content: center;
      gap: 7px;
      padding: 8px 13px;
      cursor: pointer;
      background: #fff;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      font-weight: 620;
    }
    .button:hover { background: var(--surface-muted); }
    .button.primary { color: #fff; background: var(--accent); border-color: var(--accent); }
    .button.primary:hover { background: #1d4ed8; }
    .button.danger { color: var(--danger); background: var(--danger-soft); border-color: #fecdca; }
    .result { color: var(--muted); font-size: 12px; }
    .result.success { color: var(--success); }
    .result.error { color: var(--danger); }

    .identity-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
    .search { max-width: 280px; }
    .identity-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
    .identity-card { overflow: hidden; background: #fff; border: 1px solid var(--line); border-radius: 10px; }
    .identity-card img { display: block; width: 100%; aspect-ratio: 1.18; object-fit: cover; background: #e4e7ec; }
    .identity-body { padding: 10px; }
    .identity-title { overflow: hidden; margin-bottom: 4px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .identity-detail { color: var(--muted); font-size: 11px; }
    button.identity-card { width: 100%; padding: 0; color: inherit; cursor: pointer; text-align: left; }
    button.identity-card:hover, button.identity-card:focus-visible { border-color: var(--accent); outline: 0; }
    button.identity-card[aria-expanded="true"] { border-color: var(--accent); box-shadow: inset 0 -3px var(--accent); }

    .world-entity-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
    .world-entity-card { padding: 14px; background: var(--surface-muted); border: 1px solid var(--line); cursor: pointer; transition: border-color .15s, box-shadow .15s; }
    .world-entity-card:hover { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .world-entity-card.selected { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    .world-entity-id { font-weight: 650; font-size: 13px; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 6px; }
    .world-entity-badges { display: flex; flex-wrap: wrap; gap: 5px; }
    .world-entity-badges .badge { font-size: 10px; padding: 2px 6px; }

    .world-inspector-grid { display: grid; gap: 10px; }
    .world-inspector-section { border: 1px solid var(--line); border-radius: 0; overflow: hidden; }
    .world-inspector-section summary { display: flex; cursor: pointer; list-style: none; align-items: center; justify-content: space-between; padding: 10px 14px; background: var(--surface-muted); font-weight: 620; font-size: 13px; }
    .world-inspector-section summary::-webkit-details-marker { display: none; }
    .world-inspector-section summary::after { content: "+"; color: var(--muted); }
    .world-inspector-section[open] summary::after { content: "−"; }
    .world-inspector-props { padding: 8px 14px; }
    .world-inspector-prop { display: grid; grid-template-columns: 110px 1fr; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 12px; }
    .world-inspector-prop:last-child { border-bottom: 0; }
    .world-inspector-prop-key { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; overflow-wrap: anywhere; }
    .world-inspector-prop-val { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; word-break: break-all; }
    .world-inspector-rel { display: flex; align-items: center; gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 12px; }
    .world-inspector-rel:last-child { border-bottom: 0; }
    .world-inspector-rel-arrow { color: var(--accent); font-weight: 700; }

    .person-inspector { margin-top: 16px; padding: 0; border: 1px solid var(--line-strong); background: var(--surface-muted); }
    .person-inspector-header { display: grid; grid-template-columns: 104px minmax(0, 1fr) auto; gap: 16px; align-items: center; padding: 16px; border-bottom: 1px solid var(--line); }
    .person-inspector-header > img { display: block; width: 104px; height: 104px; object-fit: cover; background: #05070a; border: 1px solid var(--line-strong); }
    .person-inspector-title { margin: 0 0 5px; color: var(--text); font-size: 18px; }
    .person-inspector-close { align-self: start; }
    .encounter-list { display: grid; }
    .encounter { display: grid; grid-template-columns: minmax(150px, .32fr) minmax(0, 1fr); border-bottom: 1px solid var(--line); }
    .encounter:last-child { border-bottom: 0; }
    .encounter-time { padding: 16px; border-right: 1px solid var(--line); }
    .encounter-date { color: var(--accent); font-size: 12px; font-weight: 700; }
    .encounter-period { margin-top: 4px; color: var(--text); font-size: 11px; }
    .encounter-evidence { display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 10px; padding: 12px; }
    .encounter-artifact { min-width: 0; padding: 8px; background: var(--surface); border: 1px solid var(--line); }
    .encounter-artifact img { display: block; width: 100%; aspect-ratio: 4 / 3; margin-bottom: 8px; object-fit: cover; background: #05070a; }
    .encounter-artifact audio { display: block; width: 100%; margin: 9px 0; }
    .encounter-artifact-time { color: var(--accent); font-size: 10px; }
    .encounter-artifact-summary { display: -webkit-box; margin-top: 5px; overflow: hidden; color: var(--muted); font-size: 11px; -webkit-box-orient: vertical; -webkit-line-clamp: 4; }
    @media (max-width: 720px) {
      .person-inspector-header { grid-template-columns: 72px minmax(0, 1fr) auto; gap: 10px; padding: 12px; }
      .person-inspector-header > img { width: 72px; height: 72px; }
      .encounter { grid-template-columns: 1fr; }
      .encounter-time { border-right: 0; border-bottom: 1px solid var(--line); }
      .encounter-evidence { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
    }
    .dream-ledger { display: grid; gap: 10px; }
    .dream-candidate { padding: 12px; background: var(--surface-muted); border: 1px solid var(--line); }
    .dream-pair { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 12px; }
    .dream-face { display: grid; grid-template-columns: 56px minmax(0, 1fr); align-items: center; gap: 10px; min-width: 0; }
    .dream-face img { width: 56px; height: 56px; object-fit: cover; border: 1px solid var(--line-strong); background: #05070a; }
    .dream-link { color: var(--accent); font-size: 18px; }

    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 10px; }
    .table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .table th, .table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
    .table th { position: sticky; top: 0; color: var(--muted); background: var(--surface-muted); font-weight: 650; }
    .table tr:last-child td { border-bottom: 0; }
    .table-button { width: 100%; cursor: pointer; color: inherit; background: transparent; border: 0; text-align: left; }

    .check-list { display: grid; gap: 2px; }
    .check { display: grid; grid-template-columns: 10px 1fr; gap: 10px; padding: 11px 2px; border-bottom: 1px solid var(--line); }
    .check:last-child { border-bottom: 0; }
    .check-dot { width: 8px; height: 8px; margin-top: 5px; background: var(--danger); border-radius: 50%; }
    .check.pass .check-dot { background: #12b76a; }
    .check.warn .check-dot { background: #f79009; }
    .check-name { font-weight: 630; }
    .check-detail { margin-top: 2px; color: var(--muted); font-size: 12px; }

    .config-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 16px; }
    .config-sections { display: grid; gap: 12px; }
    .config-section { overflow: hidden; border: 1px solid var(--line); border-radius: 11px; }
    .config-section summary { display: flex; cursor: pointer; list-style: none; align-items: center; justify-content: space-between; padding: 13px 15px; background: var(--surface-muted); font-weight: 650; }
    .config-section summary::-webkit-details-marker { display: none; }
    .config-section summary::after { content: "+"; color: var(--muted); font-size: 18px; font-weight: 400; }
    .config-section[open] summary::after { content: "−"; }
    .config-values { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 3px 15px 12px; }
    .config-row { display: grid; grid-template-columns: minmax(130px, .8fr) minmax(0, 1.2fr); gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); }
    .config-row:nth-last-child(-n+2) { border-bottom-color: transparent; }
    .config-key { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; overflow-wrap: anywhere; }
    .config-value { color: #344054; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; overflow-wrap: anywhere; }
    .config-empty { color: var(--subtle); font-style: italic; }

    /* Dark operational theme: square controls, monospace typography, amber focus. */
    ::selection { color: #07090c; background: #ffae00; }
    * { scrollbar-color: #4c3a08 #090d12; }
    .sidebar { color: #aeb6c2; border-color: var(--line); }
    .nav-label, .sidebar-meta { color: #626c79; }
    .nav-link { color: #a2aab6; }
    .nav-link:hover { color: #ffae00; background: #141820; }
    .sidebar-footer { border-color: var(--line); }
    .topbar { border-color: var(--line); }
    .mobile-menu, .card, .button, .input, .select, .identity-card, .table-wrap, .pre,
    .empty, .badge, .status-pill, .camera, .wave, .config-section, .graph-panel,
    .message { border-radius: 0; }
    .mobile-menu, .button { color: #d1d5db; background: #0d1117; border-color: var(--line-strong); }
    .mobile-menu:hover, .button:hover { color: #ffae00; background: #151a20; }
    .button.primary { color: #07090c; background: #ffae00; border-color: #ffae00; }
    .button.primary:hover { color: #07090c; background: #ffc247; }
    .button.danger { color: var(--danger); background: var(--danger-soft); border-color: #62272d; }
    .card, .identity-card { background: var(--surface); border-color: var(--line); }
    .metric-value, .card-title, .identity-title, .check-name, .page-heading h2, .page-title { color: var(--text); }
    .badge { color: #bdc4ce; background: #10151b; border-color: #303842; }
    .badge.good { color: var(--success); background: var(--success-soft); border-color: #275b35; }
    .badge.warn, .status-pill.degraded { color: #ffae00; background: #2d2205; border-color: #72550a; }
    .badge.bad, .status-pill.offline { color: var(--danger); background: var(--danger-soft); border-color: #62272d; }
    .status-pill { color: var(--success); background: var(--success-soft); border-color: #275b35; }
    .empty { background: #090d12; border-color: #303842; }
    .pre { color: #bdc4ce; background: #090d12; border-color: var(--line); }
    .input, .select { color: var(--text); background: #090d12; border-color: var(--line-strong); }
    .input:focus, .select:focus { border-color: #ffae00; box-shadow: 0 0 0 2px rgb(255 174 0 / .14); }
    .message.heard { color: #d1d5db; background: #151a20; }
    .message.agent { color: #ffca63; background: #251d08; }
    .table th { color: #9ba3af; background: #090d12; }
    .table th, .table td { border-color: var(--line); }
    .config-section summary { color: #d1d5db; background: #090d12; }
    .config-section summary::after { color: #ffae00; }
    .config-value { color: #bdc4ce; }
    .camera, .camera-head, .camera-meta { border-color: var(--line); }
    .graph-toolbar .input:focus, .graph-toolbar .select:focus { border-color: #ffae00; }

    .graph-page {
      --graph-page-width: calc(100vw - 252px);
      width: var(--graph-page-width);
      margin-left: calc((100% - var(--graph-page-width)) / 2);
    }
    .graph-page .page-heading, .graph-detail-grid, .graph-evidence-panel {
      margin-inline: clamp(18px, 3vw, 40px);
    }
    .graph-panel { width: 100%; overflow: hidden; padding: 0; background: #070d19; border-color: #1d2939; border-inline: 0; }
    .graph-toolbar {
      display: flex;
      min-height: 58px;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      color: #d0d5dd;
      background: #101828;
      border-bottom: 1px solid #344054;
    }
    .graph-toolbar-controls { display: flex; min-width: 0; align-items: center; gap: 8px; }
    .graph-toolbar-actions { display: flex; flex: 0 0 auto; align-items: center; gap: 8px; }
    .graph-toolbar .input, .graph-toolbar .select { height: 36px; color: #eaecf0; background: #182230; border-color: #344054; }
    .graph-toolbar .input::placeholder { color: #667085; }
    .graph-toolbar .button { min-height: 36px; color: #d0d5dd; background: #182230; border-color: #344054; }
    .graph-toolbar .button:hover { color: #fff; background: #253755; }
    .graph-stage { position: relative; width: 100%; height: clamp(620px, calc(100dvh - 238px), 980px); min-height: 520px; }
    .graph-canvas { position: absolute; inset: 0; overflow: hidden; cursor: grab; touch-action: none; }
    .graph-canvas:active { cursor: grabbing; }
    .graph-canvas canvas { display: block; width: 100%; height: 100%; }
    .graph-overlay { position: absolute; z-index: 2; inset: 14px auto auto 14px; pointer-events: none; }
    .graph-overlay .badge { color: #d0d5dd; background: rgb(16 24 40 / .82); border-color: #344054; backdrop-filter: blur(8px); }
    .graph-hint { position: absolute; z-index: 2; right: 14px; bottom: 12px; color: #667085; font-size: 11px; pointer-events: none; }
    .graph-detail-grid {
      display: grid;
      grid-template-columns: minmax(320px, 1.6fr) repeat(3, minmax(180px, 1fr));
      gap: 16px;
      align-items: stretch;
      margin-top: 16px;
    }
    .graph-detail-grid > .card { height: 100%; }
    .graph-selection-title { margin-bottom: 5px; font-size: 16px; font-weight: 680; overflow-wrap: anywhere; }
    .graph-selection-meta { display: grid; gap: 8px; margin-top: 14px; }
    .graph-property { display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--line); font-size: 11px; }
    .graph-property:last-child { border-bottom: 0; }
    .graph-property dt { color: var(--muted); }
    .graph-property dd { margin: 0; overflow-wrap: anywhere; }
    .graph-narrative { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
    .graph-narrative-summary { margin: 8px 0 12px; color: var(--text); line-height: 1.6; white-space: pre-wrap; }
    .graph-timeline { display: grid; gap: 8px; max-height: 420px; overflow: auto; }
    .graph-timeline-entry { padding: 8px 10px; background: var(--surface); border-left: 2px solid var(--accent); }
    .graph-timeline-time { color: var(--accent); font-size: 10px; letter-spacing: .08em; }
    .graph-timeline-summary { margin-top: 5px; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .narrative-timeline { position: relative; display: grid; gap: 0; }
    .narrative-day { display: grid; grid-template-columns: 130px 28px minmax(0, 1fr); min-width: 0; }
    .narrative-day-time { padding: 18px 16px 24px 0; color: var(--muted); font-size: 11px; text-align: right; }
    .narrative-day-time strong { display: block; color: var(--text); font-size: 13px; }
    .narrative-rail { position: relative; }
    .narrative-rail::before { position: absolute; top: 0; bottom: 0; left: 13px; width: 1px; background: var(--line); content: ''; }
    .narrative-marker { position: absolute; z-index: 1; top: 23px; left: 8px; width: 11px; height: 11px; background: var(--accent); border: 2px solid var(--bg); box-shadow: 0 0 14px rgb(255 174 0 / .55); }
    .narrative-card { min-width: 0; margin-bottom: 18px; background: var(--panel); border: 1px solid var(--line); }
    .narrative-card-button { display: block; width: 100%; padding: 16px; color: inherit; background: transparent; border: 0; text-align: left; cursor: pointer; }
    .narrative-card-button:hover { background: var(--surface); }
    .narrative-card-title { display: flex; align-items: start; justify-content: space-between; gap: 12px; }
    .narrative-card-summary { margin-top: 9px; color: var(--muted); font-size: 12px; line-height: 1.65; }
    .narrative-detail { padding: 0 16px 16px; border-top: 1px solid var(--line); }
    .narrative-periods { display: grid; gap: 10px; padding-top: 14px; }
    .narrative-period { background: var(--surface); border-left: 2px solid var(--accent); }
    .narrative-period > summary { padding: 12px; color: var(--text); cursor: pointer; }
    .narrative-period-body { padding: 0 12px 12px; }
    .narrative-artifacts { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1px; margin-top: 10px; background: var(--line); border: 1px solid var(--line); }
    .narrative-artifact { min-width: 0; padding: 10px; background: var(--panel); }
    .narrative-artifact img { display: block; width: 100%; max-height: 260px; margin-top: 8px; object-fit: contain; background: #05070a; }
    .narrative-artifact audio { display: block; width: 100%; margin-top: 8px; border-radius: 0; }
    .narrative-artifact-text { margin-top: 8px; color: var(--muted); font-size: 11px; line-height: 1.55; overflow-wrap: anywhere; }
    .narrative-episodes { display: grid; gap: 5px; margin-top: 10px; }
    .narrative-episode { padding: 8px; color: var(--muted); background: var(--panel); border: 1px solid var(--line); font-size: 10px; }
    .graph-legend { display: grid; grid-template-columns: 1fr 1fr; gap: 9px 12px; }
    .graph-evidence-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); }
    .graph-evidence-item { min-width: 0; padding: 13px; background: var(--surface); }
    .graph-evidence-item img { display: block; width: 100%; max-height: 320px; margin-top: 10px; object-fit: contain; background: #05070a; border: 1px solid var(--line); }
    .graph-evidence-item audio { display: block; width: 100%; margin-top: 10px; border-radius: 0; }
    .graph-evidence-text { max-height: 180px; overflow: auto; margin-top: 10px; padding: 9px; color: #c8cdd5; background: #090d12; border: 1px solid var(--line); white-space: pre-wrap; overflow-wrap: anywhere; }
    .graph-evidence-panel { margin-top: 16px; }
    .graph-encoding { display: grid; gap: 10px; margin: 0; }
    .graph-encoding div { display: grid; grid-template-columns: 72px minmax(0, 1fr); gap: 9px; padding-bottom: 9px; border-bottom: 1px solid var(--line); }
    .graph-encoding div:last-child { padding-bottom: 0; border-bottom: 0; }
    .graph-encoding dt { color: var(--accent); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
    .graph-encoding dd { margin: 0; color: var(--muted); font-size: 11px; }
    .graph-page.graph-theater { margin-top: -28px; }
    .graph-page.graph-theater .page-heading { display: none; }
    .graph-page.graph-theater .graph-stage { height: calc(100dvh - 130px); min-height: 600px; }
    .graph-page.graph-theater #graph-theater { color: #07090c; background: var(--accent); border-color: var(--accent); }
    .graph-panel:fullscreen { width: 100vw; height: 100vh; border: 0; background: #070d19; }
    .graph-panel:fullscreen .graph-stage { height: calc(100dvh - 58px); min-height: 0; }
    .graph-panel:fullscreen #graph-fullscreen { color: #07090c; background: var(--accent); border-color: var(--accent); }
    .legend-item { display: flex; align-items: center; gap: 8px; padding: 0; color: var(--muted); background: none; border: 0; font: inherit; font-size: 11px; text-align: left; }
    button.legend-item { cursor: pointer; }
    button.legend-item:hover, button.legend-item:focus-visible, button.legend-item.active { color: var(--text); outline: none; }
    button.legend-item:hover .legend-dot, button.legend-item:focus-visible .legend-dot, button.legend-item.active .legend-dot { box-shadow: 0 0 0 1px var(--legend), 0 0 12px var(--legend); }
    .legend-dot { width: 9px; height: 9px; flex: 0 0 auto; background: var(--legend); border-radius: 50%; box-shadow: 0 0 0 3px color-mix(in srgb, var(--legend) 18%, transparent); }

    .scrim { display: none; }
    @media (max-width: 1180px) {
      .span-3 { grid-column: span 6; }
      .span-4, .span-5, .span-7, .span-8 { grid-column: span 6; }
      .camera-grid { grid-template-columns: 1fr; }
      .config-values { grid-template-columns: 1fr; }
      .config-row:nth-last-child(-n+2) { border-bottom-color: var(--line); }
      .config-row:last-child { border-bottom-color: transparent; }
      .graph-detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 860px) {
      .sidebar { transform: translateX(-102%); transition: transform .2s ease; box-shadow: 16px 0 32px rgb(16 24 40 / .2); }
      body.menu-open .sidebar { transform: translateX(0); }
      .scrim { position: fixed; inset: 0; z-index: 35; display: block; pointer-events: none; background: rgb(16 24 40 / .5); opacity: 0; transition: opacity .2s ease; }
      body.menu-open .scrim { pointer-events: auto; opacity: 1; }
      .main { margin-left: 0; }
      .mobile-menu { display: grid; }
      .topbar { min-height: 64px; }
      .topbar-title { display: flex; align-items: center; gap: 11px; }
      .sync-label { display: none; }
      .graph-page { --graph-page-width: 100vw; }
    }
    @media (max-width: 680px) {
      .content { padding-top: 20px; }
      .page-heading { align-items: flex-start; flex-direction: column; }
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 { grid-column: 1 / -1; }
      .form-grid { grid-template-columns: 1fr; }
      .field { grid-column: 1 / -1; }
      .identity-toolbar, .config-toolbar { align-items: stretch; flex-direction: column; }
      .search { max-width: none; }
      .config-row { grid-template-columns: 1fr; gap: 3px; }
      .status-pill { max-width: 122px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .page-title { font-size: 19px; }
      .card { padding: 15px; }
      .graph-panel { padding: 0; }
      .graph-toolbar { align-items: stretch; flex-direction: column; }
      .graph-toolbar-controls { width: 100%; }
      .graph-toolbar-actions { width: 100%; flex-wrap: wrap; }
      .graph-toolbar-actions .button { flex: 1 1 auto; }
      .graph-toolbar .input { flex: 1; }
      .graph-stage { height: 68dvh; min-height: 460px; }
      .graph-detail-grid { grid-template-columns: 1fr; }
      .narrative-day { grid-template-columns: 18px minmax(0, 1fr); }
      .narrative-day-time { grid-column: 2; padding: 14px 0 8px; text-align: left; }
      .narrative-rail { grid-column: 1; grid-row: 1 / span 2; }
      .narrative-card { grid-column: 2; }
      .graph-page.graph-theater { margin-top: -20px; }
      .graph-page.graph-theater .graph-stage { height: calc(100dvh - 168px); min-height: 460px; }
      .graph-panel:fullscreen .graph-stage { height: calc(100dvh - 104px); min-height: 0; }
      .dream-pair { grid-template-columns: 1fr; }
      .dream-link { transform: rotate(90deg); text-align: center; }
    }
    @media (max-width: 480px) {
      .topbar { padding-inline: 14px; }
      .top-actions { min-width: 12px; }
      .status-pill {
        width: 12px;
        min-width: 12px;
        height: 12px;
        min-height: 12px;
        padding: 0;
        overflow: hidden;
        color: transparent;
        background: #12b76a;
        border: 0;
        box-shadow: 0 0 0 4px rgb(18 183 106 / .12);
      }
      .status-pill.degraded { background: #f79009; box-shadow: 0 0 0 4px rgb(247 144 9 / .13); }
      .status-pill.offline { background: #f04438; box-shadow: 0 0 0 4px rgb(240 68 56 / .13); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar" aria-label="Primary navigation">
      <div class="nav-label">Workspace</div>
      <nav class="nav">
        <a class="nav-link" href="/" data-route="/" data-title="Overview"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 13h6V4H4v9Zm0 7h6v-4H4v4Zm10 0h6v-9h-6v9Zm0-13h6V4h-6v3Z"/></svg><span>Overview</span></a>
        <a class="nav-link" href="/vision" data-route="/vision" data-title="Vision"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></svg><span>Vision</span></a>
        <a class="nav-link" href="/voice" data-route="/voice" data-title="Voice & Conversation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3M8 22h8"/></svg><span>Voice</span></a>
        <a class="nav-link" href="/entities" data-route="/entities" data-title="People & Objects"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="9" cy="8" r="3"/><path d="M3 20a6 6 0 0 1 12 0M16 5h5v5h-5zM17 15h4v4h-4z"/></svg><span>People & objects</span></a>
        <a class="nav-link" href="/memory" data-route="/memory" data-title="Memory"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg><span>Memory</span></a>
        <a class="nav-link" href="/cognition" data-route="/cognition" data-title="Cognition"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 4a3 3 0 0 0-5 2.2A3.5 3.5 0 0 0 5 13a4 4 0 0 0 4 6M15 4a3 3 0 0 1 5 2.2A3.5 3.5 0 0 1 19 13a4 4 0 0 1-4 6M9 4v16M15 4v16M9 8h3M12 16h3"/></svg><span>Cognition</span></a>
        <a class="nav-link" href="/graph" data-route="/graph" data-title="Knowledge graph"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="6" cy="7" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="m8.3 7 7.2-.7M7.4 9l3.5 6.8M16.6 8.2l-3.5 7.6"/></svg><span>Knowledge graph</span></a>
        <a class="nav-link" href="/dreams" data-route="/dreams" data-title="Dreams"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 15.5A8 8 0 0 1 8.5 4 8 8 0 1 0 20 15.5Z"/><path d="M15.5 5.5h3M17 4v3M5 14h3M6.5 12.5v3"/></svg><span>Dreams</span></a>
        <a class="nav-link" href="/narrative" data-route="/narrative" data-title="Narrative"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M6 3v18M6 6h12M6 12h9M6 18h12"/><circle cx="6" cy="6" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="6" cy="18" r="2"/></svg><span>Narrative</span></a>
        <a class="nav-link" href="/world" data-route="/world" data-title="World"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10A15.3 15.3 0 0 1 12 2z"/></svg><span>World</span></a>
        <a class="nav-link" href="/system" data-route="/system" data-title="System"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></svg><span>System</span></a>
      </nav>
      <div class="nav-label" style="margin-top:22px">Manage</div>
      <nav class="nav">
        <a class="nav-link" href="/configuration" data-route="/configuration" data-title="Configuration"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/></svg><span>Configuration</span></a>
      </nav>
      <div class="sidebar-footer">
        <div class="connection"><span id="connection-dot" class="connection-dot"></span><span id="connection-label">Connecting</span></div>
        <div id="sidebar-meta" class="sidebar-meta">Waiting for runtime</div>
      </div>
    </aside>
    <button id="scrim" class="scrim" aria-label="Close navigation"></button>

    <main class="main">
      <header class="topbar">
        <div class="topbar-title">
          <button id="mobile-menu" class="mobile-menu" aria-label="Open navigation"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
          <h1 id="page-title" class="page-title">Overview</h1>
        </div>
        <div class="top-actions">
          <span id="last-sync" class="sync-label">Waiting for data</span>
          <span id="runtime-status" class="status-pill degraded">Connecting</span>
        </div>
      </header>

      <div class="content">
        <section class="page" data-page="/">
          <div class="page-heading"><div><h2>Operational overview</h2><p>Current sensing, conversation, memory, and runtime health.</p></div></div>
          <div class="grid">
            <article class="card metric-card span-3"><span id="overview-runtime-dot" class="metric-indicator"></span><div class="metric-label">Runtime</div><div id="overview-runtime" class="metric-value">—</div><div id="overview-runtime-detail" class="metric-detail">Connecting</div></article>
            <article class="card metric-card span-3"><span class="metric-indicator good"></span><div class="metric-label">Camera streams</div><div id="overview-cameras" class="metric-value">0</div><div id="overview-camera-detail" class="metric-detail">No streams reported</div></article>
            <article class="card metric-card span-3"><span id="overview-asr-dot" class="metric-indicator"></span><div class="metric-label">ASR admission</div><div id="overview-asr" class="metric-value">0</div><div id="overview-asr-detail" class="metric-detail">No transcripts</div></article>
            <article class="card metric-card span-3"><span class="metric-indicator good"></span><div class="metric-label">Active episodes</div><div id="overview-episodes" class="metric-value">0</div><div id="overview-memory-detail" class="metric-detail">Memory idle</div></article>
            <article class="card span-7"><div class="card-header"><div><h3 class="card-title">Conversation</h3><p class="card-note">Latest admitted exchange</p></div><div id="overview-floor" class="badge">Listening</div></div><div id="overview-conversation" class="conversation"><div class="empty">No admitted speech yet.</div></div></article>
            <div class="span-5 stack">
              <article class="card"><div class="card-header"><div><h3 class="card-title">Scene activity</h3><p class="card-note">Labels currently present across cameras</p></div></div><div id="overview-scene" class="badge-row"><span class="muted">Waiting for vision</span></div></article>
              <article class="card"><div class="card-header"><div><h3 class="card-title">Recent decision</h3><p class="card-note">Attention and interaction policy</p></div></div><div id="overview-decision" class="muted">No decision recorded.</div></article>
            </div>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Readiness summary</h3><p class="card-note">Hardware and local service checks</p></div><button class="button" data-route-button="/system">View system</button></div><div id="overview-checks" class="badge-row"><span class="muted">Checks pending</span></div></article>
          </div>
        </section>

        <section class="page" data-page="/vision">
          <div class="page-heading"><div><h2>Camera streams</h2><p>Raw feeds with current instance masks, semantic labels, and inference timing.</p></div><div id="vision-summary" class="badge-row"></div></div>
          <div class="grid"><article class="card span-12"><div id="cameras" class="camera-grid"><div class="empty">Waiting for camera streams.</div></div></article><article class="card span-12"><div class="card-header"><div><h3 class="card-title">Scene inventory</h3><p class="card-note">Observed labels aggregated over the current runtime</p></div></div><div id="seen" class="badge-row"><span class="muted">No scene categories reported.</span></div></article></div>
          <div class="page-heading" style="margin-top:20px"><div><h2>Voxel occupancy</h2><p>Fused 3D reconstruction from the panoramic depth array; orbit to inspect the dense workable environment.</p></div><div id="occupancy-status" class="badge-row"><span class="badge">Loading</span></div></div>
          <article id="occupancy-panel" class="card graph-panel">
            <div class="graph-toolbar">
              <div class="graph-toolbar-controls"><span class="card-note">Cameras video0–video3, right to left, counter-clockwise · 60° stitch</span></div>
              <div class="graph-toolbar-actions"><button id="occupancy-voxel-scale-down" class="button" type="button" aria-label="Decrease voxel scale">Voxel −</button><button id="occupancy-voxel-scale-up" class="button" type="button" aria-label="Increase voxel scale">Voxel +</button><button id="occupancy-reset" class="button" type="button">Reset view</button></div>
            </div>
            <div class="graph-stage">
              <div id="occupancy-scene" class="graph-canvas" role="img" aria-label="Interactive three-dimensional voxel occupancy reconstruction of the environment"></div>
              <div id="occupancy-overlay" class="graph-overlay badge-row"><span class="badge">Loading occupancy</span></div>
              <div class="graph-hint">Drag to orbit · scroll to zoom</div>
            </div>
          </article>
        </section>

        <section class="page" data-page="/voice">
          <div class="page-heading"><div><h2>Voice and conversation</h2><p>Live ingress, causal turn state, transcript admission, and playback lifecycle.</p></div><div id="voice-service-state" class="badge-row"><span class="badge">Loading voice runtime</span></div></div>
          <div class="grid">
            <article class="card span-8"><div class="card-header"><div><h3 class="card-title">Audio input</h3><p id="asr-state" class="card-note">Waiting for ReSpeaker</p></div><div id="voice-floor" class="badge">Listening</div></div><canvas id="wave" class="wave"></canvas><div id="asr-metrics" class="badge-row" style="margin-top:12px"></div></article>
            <article class="card span-4"><div class="card-header"><div><h3 class="card-title">Turn lifecycle</h3><p class="card-note">Current conversation authority</p></div></div><div id="turn-state" class="table-wrap"></div></article>
            <article class="card span-7"><div class="card-header"><div><h3 class="card-title">Conversation history</h3><p class="card-note">Complete durable audible ledger; survives navigation and daemon restarts</p></div></div><div id="conversation" class="conversation"><div class="empty">No admitted speech yet.</div></div></article>
            <article class="card span-5"><div class="card-header"><div><h3 class="card-title">Live voice controls</h3><p class="card-note">Applied in-page; an Egg restart is not required</p></div><button id="voice-reload" class="button" type="button">Reload models</button></div><div id="voice-catalog-status" class="badge-row" style="margin-bottom:12px"><span class="badge">Discovering local models</span></div><form id="voice" class="form-grid"><div class="field"><label>ASR model</label><select class="select" name="asr_model"><option>Loading models…</option></select></div><div class="field"><label>ASR language</label><input class="input" name="asr_language" placeholder="en or auto" pattern="auto|[a-z]{2,3}(-[A-Z]{2})?"></div><div class="field"><label>TTS model</label><select class="select" name="voice_model"><option>Loading models…</option></select></div><div class="field"><label>Voice</label><select class="select" name="voice_name"><option>Loading voices…</option></select></div><div class="field"><label>Maximum utterance</label><input class="input" name="segment_seconds" type="number" min="1" max="15" step=".5"></div><div class="field"><label>RMS admission gate</label><input class="input" name="rms_threshold" type="number" min=".001" max="1" step=".001"></div><div class="field"><label>ASR target RMS</label><input class="input" name="asr_target_rms" type="number" min=".001" max="1" step=".001"></div><div class="field"><label>Maximum ASR gain</label><input class="input" name="asr_max_gain" type="number" min="1" max="48" step="1"></div><div class="field"><label>Pre-VAD gain</label><input class="input" name="vad_input_gain" type="number" min="1" max="32" step=".5"></div><div class="button-row"><button class="button primary" type="submit">Apply settings</button><button id="voice-reconnect" class="button" type="button">Reconnect models</button></div><span id="voice-result" class="result"></span></form></article>
          </div>
        </section>

        <section class="page" data-page="/entities">
          <div class="page-heading"><div><h2>People and objects</h2><p>Identity evidence, segmented-object learning, and review status.</p></div></div>
          <div class="grid">
            <article class="card span-12"><div class="identity-toolbar"><div><h3 class="card-title">People <span id="people-count" class="muted"></span></h3><p class="card-note">Select a person to inspect every encounter and retained artifact</p></div><input id="people-search" class="input search" type="search" placeholder="Filter people"></div><div id="identities" class="identity-grid"><div class="empty">No validated face crops yet.</div></div><section id="person-inspector" class="person-inspector" aria-live="polite" hidden></section></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Temporal person continuity</h3><p class="card-note">Adjacent mask overlap, spatial displacement, and Ornith visual comparison behind automatic single-entity tracking</p></div><div id="identity-continuity-state" class="badge-row"></div></div><div id="identity-continuity-ledger" class="table-wrap"><div class="empty">No dislocated mask merges yet.</div></div></article>
            <article class="card span-12"><div class="identity-toolbar"><div><h3 class="card-title">Segmented objects <span id="objects-count" class="muted"></span></h3><p class="card-note">Labels, confidence, provenance, and review state</p></div><input id="objects-search" class="input search" type="search" placeholder="Filter objects"></div><div id="object-learning-state" class="badge-row" style="margin-bottom:12px"></div><div id="objects" class="identity-grid"><div class="empty">No learned objects yet.</div></div></article>
          </div>
        </section>

        <section class="page" data-page="/memory">
          <div class="page-heading"><div><h2>Associative memory</h2><p>Graph entities, evidence inspection, revisions, retention, and export.</p></div><button id="export-memory" class="button">Export memory</button></div>
          <div class="grid">
            <article class="card span-7"><div class="card-header"><div><h3 class="card-title">Graph entities</h3><p id="memory-jobs" class="card-note">Waiting for memory state</p></div></div><div id="memory-stats" class="badge-row" style="margin-bottom:14px"></div><div class="table-wrap"><table class="table"><thead><tr><th>Entity</th><th>Type</th><th>State</th><th>Updated</th></tr></thead><tbody id="memory-entities"><tr><td colspan="4" class="muted">No graph entities yet.</td></tr></tbody></table></div></article>
            <article class="card span-5"><div class="card-header"><div><h3 class="card-title">Inspector</h3><p class="card-note">Select an entity from the graph</p></div></div><pre id="memory-inspector" class="pre">No entity selected.</pre></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Governance</h3><p class="card-note">Local-only mutations with explicit confirmation</p></div></div><form id="memory-controls" class="form-grid"><div class="field"><label>Entity ID</label><input class="input" name="entity_id" placeholder="person-001 or object-001"></div><div class="field"><label>Alias</label><input class="input" name="alias" placeholder="User-provided alias"></div><div class="field"><label>Claim ID</label><input class="input" name="claim_id" placeholder="Claim UUID"></div><div class="field"><label>Replacement</label><input class="input" name="replacement" placeholder="Corrected value"></div><div class="button-row"><button class="button primary" name="action" value="alias" type="submit">Add alias</button><button class="button" name="action" value="correct" type="button" id="correct-memory">Correct claim</button><button class="button danger" type="button" id="delete-memory">Delete entity</button><span id="memory-result" class="result"></span></div></form></article>
          </div>
        </section>

        <section class="page" data-page="/cognition">
          <div class="page-heading"><div><h2>Cognition</h2><p>Attention, interaction policy, retrieval, and episodic boundaries.</p></div></div>
          <div class="grid">
            <article class="card span-4"><div class="card-header"><div><h3 class="card-title">Sensing</h3><p class="card-note">Current priority target</p></div></div><div id="cognition-sensing"></div></article>
            <article class="card span-4"><div class="card-header"><div><h3 class="card-title">Decision</h3><p class="card-note">Capture and outward speech</p></div></div><div id="cognition-decision"></div></article>
            <article class="card span-4"><div class="card-header"><div><h3 class="card-title">Episodes</h3><p class="card-note">Active contexts and last boundary</p></div></div><div id="cognition-memory"></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Default-mode replay</h3><p class="card-note">Idle graph replay, reflection, and source-backed curiosity candidates</p></div></div><div id="default-mode-state"></div></article>
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Attention ledger</h3><p class="card-note">Most recent policy decisions</p></div></div><div id="attention-ledger" class="table-wrap"></div></article>
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Interaction ledger</h3><p class="card-note">Spoken and suppressed outcomes</p></div></div><div id="interaction-ledger" class="table-wrap"></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Retrieval activity</h3><p class="card-note">Evidence brought into active context</p></div></div><div id="retrieval-ledger" class="table-wrap"></div></article>
          </div>
        </section>

        <section class="page" data-page="/dreams">
          <div class="page-heading"><div><h2>Dreams</h2><p>Idle-time identity discovery, chronological replay, daily narrative consolidation, and story/meta-graph revision.</p></div><div class="button-row"><button id="dream-run" class="button primary" type="button">Dream now</button><span id="dream-result" class="result"></span></div></div>
          <div class="grid">
            <article class="card metric-card span-3"><div class="metric-label">State</div><div id="dream-state" class="metric-value">Idle</div><div id="dream-next" class="metric-detail">Schedule pending</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Canonical people</div><div id="dream-people" class="metric-value">0</div><div id="dream-fragments" class="metric-detail">No profile data</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Last pass</div><div id="dream-last-profiles" class="metric-value">—</div><div id="dream-last-detail" class="metric-detail">No completed dreams</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Retroactive merges</div><div id="dream-merges" class="metric-value">0</div><div id="dream-conflicts" class="metric-detail">Co-observation blocks: 0</div></article>
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Identity model</h3><p class="card-note">Pinned local weights; no runtime network dependency</p></div><span id="dream-model-ready" class="badge">Checking</span></div><div id="dream-model" class="pre">Loading model provenance.</div></article>
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Merge policy</h3><p class="card-note">Every automatic merge must clear all gates</p></div></div><div id="dream-policy" class="badge-row"></div><p id="dream-policy-detail" class="card-note" style="margin-top:12px"></p></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Automatic consolidation outcomes</h3><p class="card-note">Completed cluster changes and genuine safety vetoes; weak audit comparisons require no dashboard action</p></div></div><div id="dream-candidates" class="dream-ledger"><div class="empty">No dream outcomes yet.</div></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Latest chronological replay</h3><p class="card-note">Daily chapters rebuilt from ordered multimodal evidence and folded into My story</p></div><button class="button" data-route-button="/narrative">Explore narrative</button></div><div id="dream-replay"><div class="empty">No chronological dream replay yet.</div></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Dream history</h3><p class="card-note">Persisted runs, identity outcome, dated chapters replayed, story revision, and duration</p></div></div><div id="dream-history" class="table-wrap"></div></article>
          </div>
        </section>

        <section class="page" data-page="/narrative">
          <div class="page-heading"><div><h2>Narrative</h2><p>Dream-consolidated daily story, newest first, with nested encounter periods and source artifacts.</p></div><div id="narrative-status" class="badge-row"><span class="badge">Loading chapters</span></div></div>
          <article class="card"><div class="card-header"><div><h3 class="card-title">Chronological story</h3><p class="card-note">Select a day, then expand a period to inspect its summarized episodes, people, objects, speech, OCR, audio, and retained media.</p></div><button id="narrative-refresh" class="button" type="button">Replay view</button></div><div id="narrative-timeline" class="narrative-timeline"><div class="empty">Loading daily narratives…</div></div></article>
        </section>

        <section class="page" data-page="/world">
          <div class="page-heading"><div><h2>World Model</h2><p>Typed operational world model: entities, properties, relations, conflicts, and assertion history.</p></div><div id="world-status" class="badge-row"><span class="badge">Loading</span></div></div>
          <div class="grid">
            <article class="card metric-card span-3"><div class="metric-label">Entities</div><div id="world-metric-entities" class="metric-value">0</div><div id="world-metric-entities-detail" class="metric-detail">Awaiting data</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Relations</div><div id="world-metric-relations" class="metric-value">0</div><div id="world-metric-relations-detail" class="metric-detail">Awaiting data</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Conflicts</div><div id="world-metric-conflicts" class="metric-value">0</div><div id="world-metric-conflicts-detail" class="metric-detail">No conflicts</div></article>
            <article class="card metric-card span-3"><div class="metric-label">Revision</div><div id="world-metric-revision" class="metric-value">0</div><div id="world-metric-revision-detail" class="metric-detail">Latest revision</div></article>
          </div>
          <div class="grid">
            <article class="card span-8"><div class="card-header"><div><h3 class="card-title">Entities</h3><p class="card-note">Select an entity to inspect properties, relations, and assertion history.</p></div><input id="world-entity-search" class="input" type="search" placeholder="Filter entities…"></div><div id="world-entities" class="world-entity-grid"><div class="empty">Loading entities…</div></div></article>
            <article class="card span-4"><div class="card-header"><h3 class="card-title">Entity Inspector</h3></div><div id="world-entity-detail" class="card-body world-inspector-grid"><div class="empty">Select an entity above to inspect.</div></div></article>
          </div>
          <article class="card"><div class="card-header"><h3 class="card-title">Conflicts</h3></div><div id="world-conflicts" class="card-body"><div class="empty">No conflicts</div></div></article>
        </section>

        <section class="page graph-page" data-page="/graph">
          <div class="page-heading"><div><h2>Multimodal knowledge graph</h2><p>Objects, faces, recognized content, evidence, claims, and episodes arranged by relationship strength and provenance.</p></div><div id="graph-ocr-status" class="badge-row"><span class="badge">OCR awaiting data</span></div></div>
          <article id="graph-panel" class="card graph-panel">
            <div class="graph-toolbar">
              <div class="graph-toolbar-controls"><input id="graph-search" class="input" type="search" placeholder="Find an object, person, or text"><select id="graph-kind" class="select" aria-label="Filter graph modality"><option value="">All modalities</option><option value="person">People</option><option value="object">Objects</option><option value="sound_event">Sound events</option><option value="ocr_content">OCR content</option><option value="daily_narrative">Daily stories</option><option value="dream_replay">Dream replay</option><option value="world_model">World model</option><option value="evidence">Evidence</option><option value="claim">Claims</option><option value="episode">Episodes</option></select></div>
              <div class="graph-toolbar-actions"><button id="graph-theater" class="button" type="button" aria-pressed="false">Theater mode</button><button id="graph-fullscreen" class="button" type="button">Full screen</button><button id="graph-reset" class="button" type="button">Reset view</button></div>
            </div>
            <div class="graph-stage">
              <div id="knowledge-graph" class="graph-canvas" role="img" aria-label="Interactive three-dimensional graph of observed people, objects, content, and multimodal evidence"></div>
              <div id="graph-stats" class="graph-overlay badge-row"><span class="badge">Loading graph</span></div>
              <div class="graph-hint">Drag to orbit · scroll to zoom · select a node</div>
            </div>
          </article>
          <div class="graph-detail-grid">
            <article class="card graph-selection-card"><div class="card-header"><div><h3 class="card-title">Selection</h3><p class="card-note">Node and immediate relationships</p></div></div><div id="graph-selection"><div class="muted">Select a node in the graph to inspect its nested awareness and provenance.</div></div></article>
            <article class="card"><div class="card-header"><div><h3 class="card-title">Modalities</h3><p class="card-note">Hover to isolate · click to lock the filter</p></div></div><div id="graph-modality-legend" class="graph-legend"><button class="legend-item" type="button" data-graph-kind="person"><i class="legend-dot" style="--legend:#60a5fa"></i>People</button><button class="legend-item" type="button" data-graph-kind="object"><i class="legend-dot" style="--legend:#34d399"></i>Objects</button><button class="legend-item" type="button" data-graph-kind="sound_event"><i class="legend-dot" style="--legend:#ffae00"></i>Sound events</button><button class="legend-item" type="button" data-graph-kind="ocr_content"><i class="legend-dot" style="--legend:#fbbf24"></i>OCR content</button><button class="legend-item" type="button" data-graph-kind="daily_narrative"><i class="legend-dot" style="--legend:#f97316"></i>Daily stories</button><button class="legend-item" type="button" data-graph-kind="dream_replay"><i class="legend-dot" style="--legend:#8b5cf6"></i>Dream replay</button><button class="legend-item" type="button" data-graph-kind="world_model"><i class="legend-dot" style="--legend:#06b6d4"></i>World model</button><button class="legend-item" type="button" data-graph-kind="evidence"><i class="legend-dot" style="--legend:#c084fc"></i>Evidence</button><button class="legend-item" type="button" data-graph-kind="claim"><i class="legend-dot" style="--legend:#fb7185"></i>Claims</button><button class="legend-item" type="button" data-graph-kind="episode"><i class="legend-dot" style="--legend:#94a3b8"></i>Episodes</button></div></article>
            <article class="card"><div class="card-header"><div><h3 class="card-title">Live firings</h3><p class="card-note">Causal activity propagates across connected memories without moving your view</p></div></div><div class="graph-legend"><span class="legend-item"><i class="legend-dot" style="--legend:#fff"></i>Vision</span><span class="legend-item"><i class="legend-dot" style="--legend:#ffae00"></i>Heard voice</span><span class="legend-item"><i class="legend-dot" style="--legend:#c084fc"></i>Memory recall</span><span class="legend-item"><i class="legend-dot" style="--legend:#34d399"></i>Agent action</span></div></article>
            <article class="card"><div class="card-header"><div><h3 class="card-title">Relationship encoding</h3><p class="card-note">Spline form is evidence, not decoration</p></div></div><dl class="graph-encoding"><div><dt>Thickness</dt><dd>Association strength, confidence, and repeated confirmation.</dd></div><div><dt>Arch</dt><dd>Recurrence and associative separation; repeated links rise into stronger bridges.</dd></div><div><dt>Angle</dt><dd>Relationship family: identity, observation, co-presence, audio, temporal, or reflective.</dd></div><div><dt>Distance</dt><dd>Semantic affinity and confidence pull related memories into tighter neighborhoods.</dd></div></dl></article>
          </div>
          <article class="card graph-evidence-panel"><div class="card-header"><div><h3 class="card-title">Connected evidence and artifacts</h3><p class="card-note">Retained visual, audio, textual, and episodic provenance for the selected node</p></div></div><div id="graph-evidence"><div class="empty">Select a node to load its connected evidence.</div></div></article>
        </section>

        <section class="page" data-page="/system">
          <div class="page-heading"><div><h2>System</h2><p>Hardware readiness, runtime diagnostics, and compute utilization.</p></div></div>
          <div class="grid">
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Readiness</h3><p class="card-note">Direct hardware and service probes</p></div></div><div id="checks" class="check-list"><div class="empty">Checks pending.</div></div></article>
            <article class="card span-6"><div class="card-header"><div><h3 class="card-title">Runtime errors</h3><p class="card-note">Most recent component failures</p></div></div><div id="runtime-errors" class="table-wrap"></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">Cognition frequency</h3><p class="card-note">Novelty/presence/sound-driven perception rate across modalities; falls off toward an idle floor when the room is quiet, snaps back to full rate the instant something new appears or is heard</p></div><div id="activity-state" class="badge-row"></div></div><div id="activity-modalities" class="table-wrap"></div></article>
            <article class="card span-12"><div class="card-header"><div><h3 class="card-title">GPU and memory</h3><p class="card-note">jetson-stats telemetry and resident processes</p></div></div><div id="gpu-stats" class="badge-row" style="margin-bottom:14px"></div><div id="gpu-processes" class="table-wrap"></div></article>
          </div>
        </section>

        <section class="page" data-page="/configuration">
          <div class="page-heading"><div><h2>Configuration</h2><p>Complete effective runtime configuration. Voice page fields are mutable live; all other values reflect the active process.</p></div></div>
          <div class="grid">
            <article class="card span-12"><div class="config-toolbar"><div><h3 class="card-title">Effective configuration</h3><p class="card-note">All active subsystems and discovered devices</p></div><input id="config-search" class="input search" type="search" placeholder="Filter settings"></div><div id="config-sections" class="config-sections"><div class="empty">Loading configuration.</div></div></article>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
    const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
    const routeTitles = {'/':'Overview','/vision':'Vision','/voice':'Voice & Conversation','/entities':'People & Objects','/memory':'Memory','/cognition':'Cognition','/graph':'Knowledge graph','/dreams':'Dreams','/narrative':'Narrative','/world':'World','/system':'System','/configuration':'Configuration'};
    let currentState = null;
    let effectiveConfig = null;
    let catalog = null;
    let catalogLoading = false;
    let catalogRetry = null;
    let voiceFormDirty = false;
    let refreshing = false;
    let conversationLoading = false;
    let conversationLoadedAt = 0;
    let conversationLedger = [];
    let dreamLoading = false;
    let dreamLoadedAt = 0;
    let dreamState = null;
    let lastWave = [];
    let lastPeopleSignature = '';
    let lastObjectSignature = '';
    let lastContinuitySignature = '';
    let selectedPersonId = new URLSearchParams(location.search).get('person') || '';
    let personTimelineLoading = false;
    let personTimelineRevision = 0;
    let graphLoadedAt = 0;
    let graphDataSignature = '';
    let graphActivationSequence = 0;
    let occupancyLoadedAt = 0;
    let graphSelectionRevision = 0;
    let narrativeLoadedAt = 0;
    let narrativeSignature = '';
    let narrativeIndex = [];
    let narrativeDetailRevision = 0;
    const narrativeDetails = new Map();
    const cameraViews = new Map();

    function normalizedRoute(pathname) {
      const path = pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
      return routeTitles[path] ? path : '/';
    }
    function navigate(path, {replace = false} = {}) {
      const route = normalizedRoute(path);
      const priorRoute = $('.page.active')?.dataset.page;
      if (location.pathname !== route) history[replace ? 'replaceState' : 'pushState']({route}, '', route + (route === '/entities' && selectedPersonId ? `?person=${encodeURIComponent(selectedPersonId)}` : ''));
      $$('.page').forEach(page => page.classList.toggle('active', page.dataset.page === route));
      $$('.nav-link').forEach(link => {
        const active = link.dataset.route === route;
        link.classList.toggle('active', active);
        active ? link.setAttribute('aria-current', 'page') : link.removeAttribute('aria-current');
      });
      $('#page-title').textContent = routeTitles[route];
      document.title = `${routeTitles[route]} · Control Center`;
      document.body.classList.remove('menu-open');
      if (route === '/configuration' && !effectiveConfig) loadConfiguration();
      if (route === '/vision' && currentState) renderCameras(currentState.telemetry?.cameras || []);
      if (route === '/vision') loadOccupancy();
      if (route === '/vision' && priorRoute !== '/vision') window.dispatchEvent(new CustomEvent('egg:vision-activate'));
      if (priorRoute === '/vision' && route !== '/vision') window.dispatchEvent(new CustomEvent('egg:vision-deactivate'));
      if (route === '/voice' && !catalog) loadCatalog();
      if (route === '/' || route === '/voice') loadConversation();
      if (route === '/graph') loadGraph();
      if (route === '/dreams') loadDreams();
      if (route === '/narrative') loadNarratives();
      if (route === '/world') loadWorld();
      if (currentState) renderActivePage(currentState, route);
      if (priorRoute !== route) window.scrollTo({top: 0, behavior: 'instant'});
    }
    document.addEventListener('click', event => {
      const link = event.target.closest('[data-route]');
      const button = event.target.closest('[data-route-button]');
      const route = link?.dataset.route || button?.dataset.routeButton;
      if (!route || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(route);
    });
    window.addEventListener('popstate', () => {
      selectedPersonId = new URLSearchParams(location.search).get('person') || '';
      if (!selectedPersonId) { personTimelineRevision++; $('#person-inspector').hidden = true; }
      navigate(location.pathname, {replace: true});
    });
    $('#mobile-menu').addEventListener('click', () => document.body.classList.toggle('menu-open'));
    $('#scrim').addEventListener('click', () => document.body.classList.remove('menu-open'));

    function setConnection(mode, label, detail = '') {
      $('#connection-dot').className = `connection-dot ${mode}`;
      $('#connection-label').textContent = label;
      $('#sidebar-meta').textContent = detail;
      $('#runtime-status').className = `status-pill ${mode === 'online' ? '' : mode}`.trim();
      $('#runtime-status').textContent = label;
    }
    function formatTime(value) {
      if (!value) return '—';
      const date = new Date(value);
      return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    }
    function stateLabel(value) {
      const labels = {audio_detected:'Audio detected', response_playing:'Response playing', barge_pending:'Interruption pending'};
      const normalized = String(value || '');
      return labels[normalized] || normalized.replaceAll('_', ' ').replace(/^./, character => character.toUpperCase());
    }
    function statusBadge(value) {
      const normalized = String(value || 'unknown').toLowerCase();
      const tone = ['pass','ready','active','completed','true'].some(word => normalized.includes(word)) ? 'good' : ['warn','pending','degraded','interrupted'].some(word => normalized.includes(word)) ? 'warn' : ['fail','error','false'].some(word => normalized.includes(word)) ? 'bad' : '';
      return `<span class="badge ${tone}">${esc(value ?? 'unknown')}</span>`;
    }
    function table(headers, rows, emptyText = 'No records') {
      if (!rows.length) return `<div class="empty">${esc(emptyText)}</div>`;
      return `<table class="table"><thead><tr>${headers.map(header => `<th>${esc(header)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }

    function drawWave(values = lastWave) {
      lastWave = values;
      const canvas = $('#wave');
      if (!canvas) return;
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = canvas.height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, width, height);
      context.strokeStyle = '#60a5fa';
      context.lineWidth = 1.5 * ratio;
      context.beginPath();
      (values.length ? values : [0]).forEach((value, index) => {
        const x = index * width / Math.max(1, values.length - 1);
        const y = height / 2 - Number(value) * height * .42;
        index ? context.lineTo(x, y) : context.moveTo(x, y);
      });
      context.stroke();
      context.strokeStyle = 'rgba(148,163,184,.18)';
      context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
    }
    new ResizeObserver(() => drawWave()).observe($('#wave'));

    function selectOption(name, options, value) {
      const node = $(`#voice [name=${name}]`);
      const signature = JSON.stringify(options);
      const existing = node.value;
      if (node.dataset.options !== signature) {
        node.innerHTML = options.map(option => `<option value="${esc(option.id)}" ${option.disabled ? 'disabled' : ''}>${esc(option.label)}</option>`).join('');
        node.dataset.options = signature;
      }
      const desired = voiceFormDirty && options.some(option => option.id === existing) ? existing : value;
      if (document.activeElement !== node) node.value = desired ?? options[0]?.id ?? '';
    }
    function renderVoiceChoices(state) {
      if (!catalog) return;
      const voice = state.telemetry?.voice || {};
      const tts = (catalog.tts?.models || []).filter(model => model.enabled !== false).map(model => ({id:model.id,label:`${model.label || model.id} · ${model.backend || 'TTS'}`}));
      const asr = (catalog.asr?.models || []).map(model => ({id:model.id,disabled:model.liveEligible === false,label:model.id + (model.isActive ? ' · active' : '') + (model.liveEligible === false ? ` · unavailable: ${model.liveReason || 'not ready'}` : '')}));
      selectOption('voice_model', tts, voice.tts_model);
      selectOption('asr_model', asr, voice.asr_model);
      const selected = $('#voice [name=voice_model]').value;
      const voices = selected === 'supertonic' ? (catalog.supertonic?.options?.voices || []).map(id => ({id,label:id})) : [];
      selectOption('voice_name', voices, voice.tts_voice || catalog.supertonic?.settings?.voiceName);
      $('#voice [name=voice_name]').disabled = !voices.length;
      const service = catalog.state || {};
      const selectedAsr = (catalog.asr?.models || []).find(model => model.id === $('#voice [name=asr_model]').value);
      const asrReady = selectedAsr?.readiness?.weightsReady === true || service.asrReady === true;
      $('#voice-service-state').innerHTML = `<span class="badge good">Daemon online</span><span class="badge ${asrReady ? 'good' : 'warn'}">ASR ${esc($('#voice [name=asr_model]').value || 'unknown')} ${asrReady ? 'ready' : 'not ready'}</span><span class="badge ${service.voiceReady ? 'good' : 'warn'}">TTS ${esc(service.voiceModelId || $('#voice [name=voice_model]').value || 'stopped')} ${service.voiceReady ? 'ready' : 'stopped'}</span>`;
      $('#voice-catalog-status').innerHTML = `<span class="badge">${esc(asr.length)} ASR models</span><span class="badge">${esc(tts.length)} TTS models</span><span class="badge">${esc(voices.length)} voices</span>`;
    }

    function cameraView(camera) {
      let view = cameraViews.get(camera.id);
      if (view) return view;
      if (!cameraViews.size) $('#cameras').replaceChildren();
      const card = document.createElement('article'); card.className = 'camera';
      const head = document.createElement('div'); head.className = 'camera-head';
      const title = document.createElement('strong'); const rate = document.createElement('span'); rate.className = 'muted'; head.append(title, rate);
      const stage = document.createElement('div'); stage.className = 'camera-stage';
      const raw = document.createElement('img'); raw.className = 'camera-raw'; raw.alt = `${camera.id} camera stream`; raw.decoding = 'async';
      const overlay = document.createElement('div'); overlay.className = 'camera-overlay'; stage.append(raw, overlay);
      const meta = document.createElement('div'); meta.className = 'camera-head camera-meta';
      card.append(head, stage, meta); $('#cameras').append(card);
      view = {card,title,rate,stage,raw,overlay,meta}; cameraViews.set(camera.id, view); return view;
    }
    function maskSvg(detections, width, height) {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'mask-layer'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('preserveAspectRatio', 'none');
      const COCO_SKELETON = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
      const COCO_NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'];
      for (const detection of detections) {
        const polygon = detection.mask_polygon;
        if (Array.isArray(polygon) && polygon.length >= 3) {
          const points = polygon.map(point => [Number(point[0]), Number(point[1])]).filter(point => point.every(Number.isFinite));
          if (points.length >= 3) {
            const path = document.createElementNS(svg.namespaceURI, 'path'); path.setAttribute('class', 'mask'); path.setAttribute('d', `M ${points.map(point => point.join(' ')).join(' L ')} Z`); svg.append(path);
            const anchor = points.reduce((current, point) => point[1] < current[1] ? point : current, points[0]);
            const text = document.createElementNS(svg.namespaceURI, 'text'); text.setAttribute('class', 'mask-label'); text.setAttribute('x', String(anchor[0])); text.setAttribute('y', String(Math.max(18, anchor[1] - 6)));
            text.textContent = `${detection.identity || detection.label || 'object'} ${Math.round(Number(detection.identity_confidence ?? detection.confidence ?? 0) * 100)}%${detection.behavior ? ` · ${detection.behavior}` : ''}`; svg.append(text);
          }
        }
        const kps = detection.pose_keypoints;
        if (!Array.isArray(kps) || kps.length < 17) continue;
        const valid = kps.map(kp => kp && kp.length >= 3 && kp[0] > 0 && kp[1] > 0 && kp[2] > 0.25);
        for (const [a, b] of COCO_SKELETON) {
          if (!valid[a] || !valid[b]) continue;
          const line = document.createElementNS(svg.namespaceURI, 'line');
          line.setAttribute('class', 'pose-bone');
          line.setAttribute('x1', String(kps[a][0] * width)); line.setAttribute('y1', String(kps[a][1] * height));
          line.setAttribute('x2', String(kps[b][0] * width)); line.setAttribute('y2', String(kps[b][1] * height));
          svg.append(line);
        }
        for (let i = 0; i < 17; i++) {
          if (!valid[i]) continue;
          const cx = kps[i][0] * width, cy = kps[i][1] * height;
          const r = (i === 0 || i === 9 || i === 10) ? 6 : 4;
          const circle = document.createElementNS(svg.namespaceURI, 'circle');
          circle.setAttribute('class', 'pose-joint'); circle.setAttribute('cx', String(cx)); circle.setAttribute('cy', String(cy)); circle.setAttribute('r', String(r));
          svg.append(circle);
          if (i === 0 || i === 9 || i === 10) {
            const label = document.createElementNS(svg.namespaceURI, 'text');
            label.setAttribute('class', 'pose-joint-label');
            label.setAttribute('x', String(cx + 8)); label.setAttribute('y', String(cy - 6));
            label.textContent = COCO_NAMES[i]; svg.append(label);
          }
        }
      }
      return svg;
    }
    function renderCameras(cameras) {
      const present = new Set(cameras.map(camera => camera.id));
      for (const [id, view] of cameraViews) if (!present.has(id)) { view.card.remove(); cameraViews.delete(id); }
      if (!cameras.length) { $('#cameras').innerHTML = '<div class="empty">No camera streams available.</div>'; return; }
      for (const camera of cameras) {
        const view = cameraView(camera), shape = Array.isArray(camera.frame_shape) ? camera.frame_shape : [], height = Number(shape[0]), width = Number(shape[1]);
        view.title.textContent = camera.id; view.rate.textContent = `${camera.fps ?? '—'} stream · ${camera.inference_fps ?? '—'} inference FPS`;
        view.stage.style.aspectRatio = width > 0 && height > 0 ? `${width}/${height}` : '16/9';
        if (camera.raw_stream_url && view.raw.dataset.stream !== camera.raw_stream_url) { view.raw.src = camera.raw_stream_url; view.raw.dataset.stream = camera.raw_stream_url; }
        view.overlay.replaceChildren(maskSvg(camera.detections || [], width || 16, height || 9));
        const labels = document.createElement('div'); labels.className = 'badge-row';
        for (const label of camera.semantic_labels || []) { const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = label; labels.append(badge); }
        if (!labels.childElementCount) { const none = document.createElement('span'); none.className = 'muted'; none.textContent = 'Inference pending'; labels.append(none); }
        const stamp = document.createElement('span'); stamp.className = 'muted'; stamp.textContent = camera.detections_updated_at ? `#${camera.detection_sequence || 0} · ${formatTime(camera.detections_updated_at)}` : 'Pending';
        view.meta.replaceChildren(labels, stamp);
      }
    }

    function renderConversation(telemetry, target, full = false) {
      const node = $(target);
      const durable = conversationLedger;
      let turns = full ? durable : durable.slice(-4);
      if (!turns.length) {
        turns = [
          telemetry.latest_transcript ? {role:'heard', text:telemetry.latest_transcript, status:'final', at:telemetry.latest_transcript_at} : null,
          telemetry.latest_reply ? {role:'agent', text:telemetry.latest_reply, status:'spoken'} : null,
        ].filter(Boolean);
      }
      const signature = JSON.stringify(turns.map(turn => [turn.id,turn.role,turn.text,turn.status,turn.at,turn.tags,turn.tool_calls]));
      if (node.dataset.signature === signature) return;
      const pinnedToBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 36;
      const previousScroll = node.scrollTop;
      node.dataset.signature = signature;
      if (!turns.length) { node.innerHTML = '<div class="empty">No admitted speech yet.</div>'; return; }
      node.innerHTML = turns.map(turn => {
        const tags = Array.isArray(turn.tags) ? turn.tags.slice(0, 16) : [];
        const tagMarkup = tags.length ? `<span class="message-tags">${tags.map(tag => `<span class="message-tag ${esc(String(tag.kind || 'modality'))}" title="${esc(String(tag.kind || 'evidence'))}">${esc(String(tag.label || 'evidence'))}</span>`).join('')}</span>` : '';
        return `<div class="message ${turn.role === 'agent' ? 'agent' : 'heard'} ${turn.status === 'suppressed' ? 'suppressed' : ''}"><span class="message-role">${turn.role === 'agent' ? 'Egg' : 'Heard'}</span>${esc(turn.text)}${tagMarkup}<span class="message-meta">${turn.at ? esc(new Date(turn.at).toLocaleString()) + ' · ' : ''}${esc(stateLabel(turn.status || 'final'))}</span></div>`;
      }).join('');
      node.scrollTop = pinnedToBottom ? node.scrollHeight : previousScroll;
    }
    async function loadConversation(force = false) {
      const now = Date.now();
      if (conversationLoading || (!force && now - conversationLoadedAt < 4000)) return;
      conversationLoading = true;
      try {
        const response = await fetch('/api/voice/conversation?limit=5000', {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const ledger = await response.json();
        if (Array.isArray(ledger)) conversationLedger = ledger;
        conversationLoadedAt = Date.now();
        const telemetry = currentState?.telemetry || {};
        if ($('.page.active')?.dataset.page === '/') renderConversation(telemetry, '#overview-conversation');
        if ($('.page.active')?.dataset.page === '/voice') renderConversation(telemetry, '#conversation', true);
      } catch (_) {
        conversationLoadedAt = 0;
      } finally {
        conversationLoading = false;
      }
    }
    function renderOverview(state) {
      const telemetry = state.telemetry || {}, cameras = telemetry.cameras || [], asr = telemetry.asr || {}, memory = telemetry.memory?.lifecycle || {}, active = memory.active || [], checks = state.checks || [];
      const runtime = String(state.runtime || 'unknown'); const degraded = runtime.includes('degraded');
      $('#overview-runtime').textContent = degraded ? 'Degraded' : runtime.includes('live') ? 'Online' : 'Starting';
      $('#overview-runtime-detail').textContent = degraded ? 'One or more checks require attention' : 'Runtime services active';
      $('#overview-runtime-dot').className = `metric-indicator ${degraded ? 'warn' : 'good'}`;
      $('#overview-cameras').textContent = cameras.length; $('#overview-camera-detail').textContent = cameras.length ? `${cameras.reduce((sum, camera) => sum + (camera.detections || []).length, 0)} current detections` : 'No streams reported';
      $('#overview-asr').textContent = telemetry.transcript_count || 0; $('#overview-asr-detail').textContent = `${asr.accepted || 0} accepted · ${asr.rejected || 0} rejected`;
      $('#overview-asr-dot').className = `metric-indicator ${asr.errors ? 'warn' : 'good'}`;
      $('#overview-episodes').textContent = active.length; $('#overview-memory-detail').textContent = `${telemetry.memory?.accepted_events || 0} accepted events`;
      $('#overview-floor').textContent = telemetry.voice?.floor || 'listening'; renderConversation(telemetry, '#overview-conversation');
      $('#overview-scene').innerHTML = (telemetry.seen || []).slice(0, 16).map(item => `<span class="badge">${esc(item.label)} <strong>×${esc(item.count)}</strong></span>`).join('') || '<span class="muted">No scene labels reported.</span>';
      const attention = (telemetry.attention_decisions || []).at(-1), interaction = (telemetry.interaction_decisions || []).at(-1);
      $('#overview-decision').innerHTML = `<strong>${esc(attention?.reason || 'No attention decision')}</strong><div class="card-note" style="margin-top:7px">${esc(interaction?.allowed ? 'Spoken' : 'Suppressed')} · ${esc(interaction?.reason || 'No interaction decision')}</div>`;
      $('#overview-checks').innerHTML = checks.slice(0, 12).map(check => `<span class="badge ${check.status === 'pass' ? 'good' : check.status === 'warn' ? 'warn' : 'bad'}">${esc(check.name)}</span>`).join('') || '<span class="muted">Checks pending.</span>';
    }
    function renderVoice(telemetry) {
      const voice = telemetry.voice || {}, vad = telemetry.vad || {}, asr = telemetry.asr || {}, mic = telemetry.respeaker || {}, comprehension = telemetry.audio_comprehension || {};
      $('#asr-state').textContent = `${voice.asr_input || 'ASR input'} · ${vad.speech ? 'speech' : 'silence'} · ${Math.round(Number(vad.speech_ratio || 0) * 100)}% / ${Number(vad.speech_ms || 0)} ms · RMS ${Number(telemetry.audio_rms || 0).toFixed(4)}`;
      $('#voice-floor').textContent = stateLabel(voice.floor || 'listening');
      $('#asr-metrics').innerHTML = `<span class="badge good">${esc(asr.accepted || 0)} accepted</span><span class="badge">${esc(asr.rejected || 0)} rejected</span><span class="badge ${asr.errors ? 'bad' : ''}">${esc(asr.errors || 0)} errors</span><span class="badge ${comprehension.state === 'completed' ? 'good' : comprehension.state === 'error' ? 'bad' : ''}">Audio comprehension ${esc(stateLabel(comprehension.state || 'idle'))}</span><span class="badge">${esc(comprehension.completed || 0)} semantic passes</span><span class="badge ${mic.ready ? 'good' : 'warn'}">XVF3000 ${mic.ready ? 'online' : 'pending'}</span><span class="badge">DoA ${mic.doa_angle == null ? '—' : Math.round(Number(mic.doa_angle)) + '°'}</span><span class="badge ${mic.voice_activity ? 'good' : ''}">DSP VAD ${mic.voice_activity ? 'voice' : 'quiet'}</span>${asr.last_rejection ? `<span class="badge warn">${esc(asr.last_rejection)}</span>` : ''}`;
      const rows = [['Floor', stateLabel(voice.floor || '—')], ['Revision', voice.revision ?? '—'], ['Pending ingress', voice.pending_ingress ?? 0], ['Playback', stateLabel(voice.playback_status || 'none')], ['Interruption', voice.active_barge_id ? 'Pending' : 'None'], ['ReSpeaker LEDs', stateLabel(mic.led_state || 'pending')], ['AEC far end', mic.aec_far_end_silence == null ? '—' : mic.aec_far_end_silence ? 'silent' : 'active'], ['AGC gain', mic.agc_gain == null ? '—' : Number(mic.agc_gain).toFixed(2) + '×'], ['Room RT60', mic.rt60_seconds == null ? '—' : Number(mic.rt60_seconds).toFixed(2) + ' s'], ['Last transition', stateLabel(voice.last_transition_reason || '—')]];
      $('#turn-state').innerHTML = table(['State','Value'], rows.map(row => [esc(row[0]), `<span class="mono">${esc(row[1])}</span>`]));
      renderConversation(telemetry, '#conversation', true);
      for (const [key, value] of Object.entries({segment_seconds:voice.asr_segment_seconds, rms_threshold:voice.asr_rms_threshold, asr_target_rms:voice.asr_target_rms, asr_max_gain:voice.asr_max_gain, vad_input_gain:voice.vad_input_gain, asr_language:voice.asr_language})) { const input = $(`#voice [name=${key}]`); if (!voiceFormDirty && document.activeElement !== input) input.value = value ?? ''; }
      renderVoiceChoices({telemetry});
    }
    function renderEntities(state) {
      const people = state.identities || [], objects = state.objects || [], learning = state.telemetry?.object_learning || {};
      const identity = state.identity_summary || {}, continuity = state.telemetry?.identity_continuity || {};
      $('#people-count').textContent = `(${identity.named_people || 0} named · ${identity.recurrent_face_profiles || 0} recurrent · ${identity.provisional_face_profiles || 0} provisional)`; $('#objects-count').textContent = `(${objects.length})`;
      const peopleQuery = $('#people-search').value.trim().toLowerCase();
      const visiblePeople = people.filter(item => `${item.label || ''} ${item.id || ''}`.toLowerCase().includes(peopleQuery)).slice(0, 48);
      const peopleSignature = JSON.stringify(visiblePeople.map(item => [item.id,item.label,item.confidence,item.samples,item.sightings,item.last_seen,item.status,item.stack_count]));
      if (peopleSignature !== lastPeopleSignature) { lastPeopleSignature = peopleSignature; $('#identities').innerHTML = visiblePeople.map(identity => `<button class="identity-card" type="button" data-person-id="${esc(identity.id)}" aria-expanded="${identity.id === selectedPersonId}"><img src="${esc(identity.thumbnail_url)}?t=${encodeURIComponent(identity.last_seen || '')}" alt="${esc(identity.label || 'Person')} evidence crop"><div class="identity-body"><div class="identity-title">${esc(identity.label || identity.id)}</div><div class="identity-detail">${esc(identity.status || identity.kind || 'face')} · ${Math.round(Number(identity.confidence || 0) * 100)}%<br>${Number(identity.stack_count || 1) > 1 ? `${esc(identity.stack_count)} profiles stacked · ` : ''}${esc(identity.samples || 0)} retained samples · ${esc(identity.sightings || 0)} historical observations<br><span style="color:var(--accent)">Open encounter history →</span></div></div></button>`).join('') || '<div class="empty">No matching people.</div>'; }
      const objectQuery = $('#objects-search').value.trim().toLowerCase(); const visibleObjects = objects.filter(item => `${item.label || ''} ${item.id || ''}`.toLowerCase().includes(objectQuery)).slice(0, 48);
      const objectSignature = JSON.stringify(visibleObjects.map(item => [item.id,item.label,item.confidence,item.samples,item.last_seen,item.review_state]));
      if (objectSignature !== lastObjectSignature) { lastObjectSignature = objectSignature; $('#objects').innerHTML = visibleObjects.map(object => `<article class="identity-card"><img src="${esc(object.thumbnail_url)}?t=${encodeURIComponent(object.last_seen || '')}" alt="${esc(object.label || 'Object')} crop"><div class="identity-body"><div class="identity-title">${esc(object.label || object.id)}</div><div class="identity-detail">${Math.round(Number(object.label_confidence || object.confidence || 0) * 100)}% · ${esc(object.samples || 0)} samples<br>${esc(object.last_match_state || object.review_state || 'pending')} · ${esc(object.label_source || 'unknown')}</div></div></article>`).join('') || '<div class="empty">No matching objects.</div>'; }
      const continuitySignature = JSON.stringify(continuity);
      if (continuitySignature !== lastContinuitySignature) {
        lastContinuitySignature = continuitySignature;
        $('#identity-continuity-state').innerHTML = `<span class="badge ${continuity.state === 'error' ? 'bad' : continuity.completed ? 'good' : ''}">${esc(stateLabel(continuity.state || 'idle'))}</span><span class="badge">${esc(continuity.completed || 0)} analyzed</span><span class="badge">${esc(continuity.queued || 0)} queued</span><span class="badge ${continuity.disagreements ? 'warn' : ''}">${esc(continuity.disagreements || 0)} VLM disagreements</span><span class="badge ${continuity.errors ? 'bad' : ''}">${esc(continuity.errors || 0)} errors</span>`;
        const continuityRows = (continuity.recent || []).slice().reverse().map(item => {
          const geometry = item.geometry || {}, analysis = item.analysis || {};
          return [
            `<strong>${esc(item.entity_id || 'person')}</strong><div class="muted mono">${esc(item.camera_id || 'camera')}</div>`,
            `mask IoU ${Number(geometry.mask_iou || 0).toFixed(3)} · containment ${Number(geometry.mask_containment || 0).toFixed(3)}<br><span class="muted">Δ ${Number(geometry.centroid_dx_pixels || 0).toFixed(1)}, ${Number(geometry.centroid_dy_pixels || 0).toFixed(1)} px · ${Number(geometry.elapsed_seconds || 0).toFixed(2)} s</span>`,
            `${statusBadge(analysis.same_person === false ? 'review' : 'consolidated')} ${esc(analysis.analysis || item.detail || 'analysis pending')}`,
            esc(analysis.displacement_analysis || 'No VLM displacement narrative'),
            esc(formatTime(item.at)),
          ];
        });
        $('#identity-continuity-ledger').innerHTML = table(['Entity','Mask / displacement','VLM comparison','Displacement analysis','Time'], continuityRows, 'No dislocated mask merges yet');
      }
      $('#object-learning-state').innerHTML = `<span class="badge">${esc(learning.stable_candidates || 0)} stable</span><span class="badge">${esc(learning.clip_recalls || 0)}/${esc(learning.clip_queries || 0)} CLIP</span><span class="badge">${esc(learning.vlm_successes || 0)}/${esc(learning.vlm_requests || 0)} VLM</span><span class="badge">${esc(learning.ocr_hits || 0)}/${esc(learning.ocr_requests || 0)} OCR</span><span class="badge ${learning.vlm_errors ? 'bad' : ''}">${esc(learning.vlm_errors || 0)} errors</span><span class="badge ${learning.review_queue_depth ? 'warn' : 'good'}">${esc(learning.review_queue_depth || 0)} awaiting re-ID</span>`;
      if (selectedPersonId && $('#person-inspector').hidden && !personTimelineLoading) loadPersonTimeline(selectedPersonId);
    }
    function timelineRange(encounter) {
      const start = new Date(encounter.started_at), end = new Date(encounter.ended_at);
      const time = value => value.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
      return start.getTime() === end.getTime() ? time(start) : `${time(start)}–${time(end)}`;
    }
    function renderPersonTimeline(timeline) {
      const inspector = $('#person-inspector'), encounters = timeline.encounters || [];
      inspector.hidden = false;
      inspector.innerHTML = `<header class="person-inspector-header"><img src="${esc(timeline.thumbnail_url)}?t=${encodeURIComponent(timeline.last_seen || '')}" alt="${esc(timeline.label)} canonical identity"><div><h3 class="person-inspector-title">${esc(timeline.label)}</h3><div class="badge-row"><span class="badge good">${esc(timeline.encounter_count || 0)} encounter periods</span><span class="badge">${esc(timeline.evidence_event_count || 0)} evidence events</span><span class="badge">${esc(timeline.retained_artifact_count || 0)} artifacts</span><span class="badge">${esc(timeline.source_profile_ids?.length || 1)} profiles consolidated</span></div><p class="card-note">Full history ${esc(new Date(timeline.first_seen).toLocaleString())} → ${esc(new Date(timeline.last_seen).toLocaleString())}</p></div><button id="person-inspector-close" class="button person-inspector-close" type="button" aria-label="Close encounter history">Close</button></header><div class="encounter-list">${encounters.map(encounter => `<article class="encounter"><div class="encounter-time"><div class="encounter-date">${esc(new Date(encounter.started_at).toLocaleDateString([], {weekday:'short', year:'numeric', month:'short', day:'numeric'}))}</div><div class="encounter-period">${esc(timelineRange(encounter))}</div><div class="card-note">${esc(encounter.event_count)} events · ${esc((encounter.modalities || []).join(' + '))}<br>${esc((encounter.sources || []).join(' · ') || 'local sensor')}</div></div><div class="encounter-evidence">${(encounter.events || []).map(event => { const media = event.artifact_url && event.artifact_kind === 'audio' ? `<audio controls preload="metadata" src="${esc(event.artifact_url)}"></audio>` : event.artifact_url ? `<img loading="lazy" src="${esc(event.artifact_url)}" alt="Evidence captured ${esc(new Date(event.captured_at).toLocaleString())}">` : ''; return `<div class="encounter-artifact">${media}<div class="encounter-artifact-time">${esc(new Date(event.captured_at).toLocaleTimeString())} · ${esc(event.modality)}</div><div class="encounter-artifact-summary">${esc(event.summary || 'Observed')}<br>${esc(event.source || '')}</div></div>`; }).join('') || '<div class="empty">No retained artifact for this encounter.</div>'}</div></article>`).join('') || '<div class="empty">No historical encounter evidence is retained yet.</div>'}</div>`;
      $$('#identities [data-person-id]').forEach(card => card.setAttribute('aria-expanded', String(card.dataset.personId === selectedPersonId)));
    }
    async function loadPersonTimeline(profileId, {focus = false} = {}) {
      selectedPersonId = profileId; personTimelineLoading = true; const revision = ++personTimelineRevision, inspector = $('#person-inspector');
      inspector.hidden = false; inspector.innerHTML = '<div class="empty">Loading complete encounter history…</div>';
      history.replaceState({route:'/entities', person:profileId}, '', `/entities?person=${encodeURIComponent(profileId)}`);
      $$('#identities [data-person-id]').forEach(card => card.setAttribute('aria-expanded', String(card.dataset.personId === profileId)));
      try { const response = await fetch(`/api/identities/${encodeURIComponent(profileId)}/timeline`, {cache:'no-store'}); if (!response.ok) throw new Error(await response.text()); if (revision === personTimelineRevision) renderPersonTimeline(await response.json()); }
      catch (error) { if (revision === personTimelineRevision) inspector.innerHTML = `<div class="empty">Encounter history unavailable: ${esc(error.message)}</div>`; }
      finally { if (revision === personTimelineRevision) personTimelineLoading = false; if (focus) inspector.scrollIntoView({behavior:'smooth', block:'start'}); }
    }
    function renderMemory(memory) {
      const stats = memory?.stats || {}, entities = memory?.entities || [], jobs = memory?.jobs || [], conflicts = memory?.claim_conflicts || [], buffer = memory?.transient_buffer || {};
      $('#memory-stats').innerHTML = Object.entries(stats).map(([key,value]) => `<span class="badge">${esc(key.replaceAll('_',' '))} <strong>${esc(value)}</strong></span>`).join('') || '<span class="muted">Memory unavailable.</span>';
      $('#memory-entities').innerHTML = entities.slice(0, 150).map(entity => `<tr><td><button class="table-button" type="button" data-entity="${esc(entity.entity_id)}"><strong>${esc(entity.display_name || entity.entity_id)}</strong><div class="muted mono">${esc(entity.entity_id)}</div></button></td><td>${esc(entity.entity_type)}</td><td>${statusBadge(entity.state)}</td><td>${esc(formatTime(entity.updated_at))}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">No graph entities yet.</td></tr>';
      $('#memory-jobs').textContent = `Jobs ${jobs.length ? jobs[0].state : 'none'} · ${conflicts.length} unresolved conflicts · ${(buffer.frame_references || 0) + (buffer.audio_references || 0)} transient references`;
    }
    function renderCognition(telemetry) {
      const brain = telemetry.brain || {}, sensing = brain.sensing || {}, cognition = brain.cognition || {}, lifecycle = telemetry.memory?.lifecycle || {}, active = lifecycle.active || [], boundary = lifecycle.last_boundary || {};
      const defaultMode = telemetry.default_mode || {}, graphFeedback = brain.graph_feedback || {}, observationPolicy = brain.observation_policy || {};
      $('#cognition-sensing').innerHTML = `<div class="metric-value">${esc(sensing.target_count || 0)}</div><div class="metric-detail">targets · ${esc(sensing.top_label || 'none')}</div><div class="badge-row" style="margin-top:12px"><span class="badge">priority ${sensing.top_priority == null ? '—' : Math.round(sensing.top_priority * 100) + '%'}</span><span class="badge">novelty ${esc(sensing.novelty || 0)}</span></div>`;
      $('#cognition-decision').innerHTML = `<div class="metric-value">${cognition.allow_outward_speech ? 'External' : 'Internal'}</div><div class="metric-detail">${esc(cognition.reason || 'idle')}</div><div class="badge-row" style="margin-top:12px"><span class="badge">capture ${cognition.capture_priority == null ? '—' : Math.round(cognition.capture_priority * 100) + '%'}</span></div>`;
      $('#cognition-memory').innerHTML = `<div class="metric-value">${active.length}</div><div class="metric-detail">active contexts</div><div class="badge-row" style="margin-top:12px"><span class="badge">${esc(boundary.reason || 'no boundary')}</span></div>`;
      $('#default-mode-state').innerHTML = `<div class="badge-row"><span class="badge ${defaultMode.state === 'failed' ? 'bad' : defaultMode.state === 'complete' ? 'good' : ''}">${esc(stateLabel(defaultMode.state || 'idle'))}</span><span class="badge">${esc((defaultMode.replayed_entity_ids || []).length)} provenance entities replayed</span><span class="badge">${esc(Object.keys(graphFeedback).length)} live graph feedback signals</span><span class="badge ${observationPolicy.state === 'model_complete' ? 'good' : ''}">${esc(stateLabel(observationPolicy.state || 'pending model semantics'))}</span></div>${(observationPolicy.focus_terms || []).length ? `<div class="badge-row" style="margin-top:12px">${observationPolicy.focus_terms.slice(0,12).map(term => `<span class="badge good">model focus: ${esc(term)}</span>`).join('')}${(observationPolicy.open_questions || []).length ? `<span class="badge warn">${esc(observationPolicy.open_questions.length)} model-authored open threads</span>` : ''}</div>` : ''}`;
      const attention = (telemetry.attention_decisions || []).slice(-12).reverse(); $('#attention-ledger').innerHTML = table(['Reason','Priority','Time'], attention.map(item => [esc(item.reason || '—'), esc(item.capture_priority ?? '—'), esc(formatTime(item.at))]), 'No attention decisions');
      const interactions = (telemetry.interaction_decisions || []).slice(-12).reverse(); $('#interaction-ledger').innerHTML = table(['Outcome','Reason','Time'], interactions.map(item => [statusBadge(item.allowed ? 'spoken' : 'suppressed'), esc(item.reason || '—'), esc(formatTime(item.at))]), 'No interaction decisions');
      const retrieval = (telemetry.retrieval_hits || []).slice(-16).reverse(); $('#retrieval-ledger').innerHTML = table(['Source','Detail','Score'], retrieval.map(item => [esc(item.source || item.kind || 'evidence'), esc(item.detail || item.reason || item.entity_id || '—'), esc(item.score ?? item.confidence ?? '—')]), 'No retrieval activity');
    }
    function renderDreams(dreams, identity) {
      dreams = dreams || {}; identity = identity || {};
      const runs = dreams.runs || [], candidates = dreams.candidates || [], last = runs[0] || {}, model = dreams.model || {}, policy = dreams.policy || {};
      const lastReplay = dreams.narrative_replay || last.details?.chronological_replay || {};
      $('#dream-state').textContent = stateLabel(dreams.state || (dreams.enabled ? 'idle' : 'disabled'));
      $('#dream-next').textContent = dreams.next_scheduled_at ? `Next idle window ${new Date(dreams.next_scheduled_at).toLocaleString()}` : 'Schedule pending';
      $('#dream-people').textContent = identity.canonical_people || 0;
      $('#dream-fragments').textContent = `${identity.coalesced_aliases || 0} aliases · ${identity.provisional_face_profiles || 0} provisional · ${identity.quarantined_face_samples || 0} invalid crops quarantined`;
      $('#dream-last-profiles').textContent = last.profiles_examined ?? '—';
      $('#dream-last-detail').textContent = last.started_at || lastReplay.replayed_at ? `${last.samples_embedded || 0} face crops · ${lastReplay.days_replayed || 0} day chapters · story r${lastReplay.story_revision ?? '—'} · ${Number(last.duration_seconds || 0).toFixed(2)} s` : 'No completed dreams';
      $('#dream-merges').textContent = identity.coalesced_aliases || 0;
      $('#dream-conflicts').textContent = `Last pass: ${last.merges || 0} merged · ${lastReplay.days_replayed || 0} days replayed · ${last.conflicts_blocked || 0} safety vetoes`;
      $('#dream-model-ready').className = `badge ${model.ready ? 'good' : 'bad'}`;
      $('#dream-model-ready').textContent = model.ready ? `${esc(model.configured_device || model.device || 'local')} ready` : 'Weights unavailable';
      $('#dream-model').textContent = `${model.architecture || 'AdaFace IR18'}\n${model.id || 'model unavailable'}\nrevision ${model.revision || '—'}\nconfigured ${model.configured_device || '—'} · runtime ${model.device || 'unloaded'}\n${model.path || '—'}\n\ncomparison ${model.comparison?.id || 'not configured'} · ${model.comparison?.ready ? (model.comparison?.device || 'ready') : 'unavailable'}\n${model.comparison?.path || '—'}\n\n${model.usage_notice || ''}`;
      $('#dream-policy').innerHTML = (policy.constraints || []).map(item => `<span class="badge">${esc(item)}</span>`).join('') || '<span class="muted">Policy unavailable.</span>';
      $('#dream-policy-detail').textContent = `Idle ${policy.idle_seconds ?? '—'} s · randomized interval ${policy.interval_min_seconds ?? '—'}–${policy.interval_max_seconds ?? '—'} s · ${policy.minimum_model_votes ?? 2}-model consensus · AdaFace ${policy.modern_merge_similarity ?? '—'} / SFace ${policy.legacy_merge_similarity ?? '—'} / MobileFaceNet ${policy.comparison_merge_similarity ?? '—'} · ${policy.coobservation_min_confirmations ?? '—'} co-observations to veto`;
      const latestCandidates = candidates.filter(item => !last.run_id || item.run_id === last.run_id), auditOnly = latestCandidates.filter(item => item.decision === 'review').length;
      const recent = latestCandidates.filter(item => item.decision !== 'review').slice(0, 60);
      $('#dream-candidates').innerHTML = recent.map(item => {
        const tone = ['merged','consolidated'].includes(item.decision) ? 'good' : item.decision === 'blocked' ? 'bad' : 'warn';
        return `<article class="dream-candidate"><div class="dream-pair"><div class="dream-face"><img loading="lazy" src="/api/identities/${encodeURIComponent(item.left_id)}/face.jpg" alt="Evidence for ${esc(item.left_id)}"><div><strong>${esc(item.left_id)}</strong><div class="muted">AdaFace ${Number(item.modern_similarity || 0).toFixed(3)}</div></div></div><div class="dream-link">↔</div><div class="dream-face"><img loading="lazy" src="/api/identities/${encodeURIComponent(item.right_id)}/face.jpg" alt="Evidence for ${esc(item.right_id)}"><div><strong>${esc(item.right_id)}</strong><div class="muted">SFace ${Number(item.legacy_similarity || 0).toFixed(3)} · MobileFaceNet ${item.comparison_similarity == null ? '—' : Number(item.comparison_similarity).toFixed(3)}</div></div></div></div><div class="badge-row" style="margin-top:10px"><span class="badge ${tone}">${esc(item.decision)}</span><span class="badge">${esc(item.reason || 'evaluated')}</span><span class="badge">margins ${Number(item.left_margin || 0).toFixed(3)} / ${Number(item.right_margin || 0).toFixed(3)}</span>${item.canonical_id ? `<span class="badge good">canonical ${esc(item.canonical_id)}</span>` : ''}</div></article>`;
      }).join('') + (auditOnly ? `<div class="empty">${esc(auditOnly)} weak comparisons retained in the audit database; they are not stalled work and require no review.</div>` : '') || '<div class="empty">No identity changes or safety vetoes in the latest dream.</div>';
      if (lastReplay.state === 'failed') {
        $('#dream-replay').innerHTML = `<div class="empty">Chronological replay failed: ${esc(lastReplay.error || 'unknown error')}</div>`;
      } else if (lastReplay.days_replayed) {
        const chapters = Array.isArray(lastReplay.daily_narratives) ? lastReplay.daily_narratives : [];
        $('#dream-replay').innerHTML = `<div class="badge-row"><span class="badge good">${esc(lastReplay.days_replayed)} days replayed</span><span class="badge">${esc(lastReplay.backfilled_days?.length || 0)} backdated</span><span class="badge ${lastReplay.backlog_remaining ? 'warn' : 'good'}">${esc(lastReplay.backlog_remaining || 0)} awaiting narrative</span><span class="badge">My story r${esc(lastReplay.story_revision ?? '—')}</span><span class="badge">${esc(lastReplay.meta_graph?.documents_revised || 0)} documents revised</span><span class="badge">${esc(lastReplay.meta_graph?.abstractions_projected || 0)} abstractions active</span></div><div class="dream-ledger" style="margin-top:12px">${chapters.map(chapter => `<article class="dream-candidate"><strong>${esc(chapter.local_date || 'dated chapter')}</strong><div class="card-note" style="margin-top:6px">${esc(chapter.abstract_summary || 'Chronological evidence replayed.')}</div><div class="badge-row" style="margin-top:8px"><span class="badge">r${esc(chapter.revision || 0)}</span><span class="badge">${esc(chapter.timeline_entries || 0)} periods</span><span class="badge">${esc(chapter.evidence_count || 0)} artifacts</span><span class="badge">${esc(chapter.episode_count || 0)} episodes</span></div></article>`).join('')}</div>`;
      } else {
        $('#dream-replay').innerHTML = '<div class="empty">No retained day was available for chronological replay.</div>';
      }
      $('#dream-history').innerHTML = table(['Started','State','Device','Profiles / samples','Proposals','Merges','Day chapters / story','Duration'], runs.map(run => { const replay = run.details?.chronological_replay || {}; return [esc(new Date(run.started_at).toLocaleString()), statusBadge(run.state), esc(run.device || '—'), `${esc(run.profiles_examined || 0)} / ${esc(run.samples_embedded || 0)}`, esc(run.proposals || 0), esc(run.merges || 0), replay.state === 'failed' ? '<span class="badge bad">replay failed</span>' : `${esc(replay.days_replayed || 0)} / r${esc(replay.story_revision ?? '—')}`, `${Number(run.duration_seconds || 0).toFixed(2)} s`]; }), 'No dream runs recorded');
    }
    async function loadDreams(force = false) {
      if (dreamLoading || (!force && Date.now() - dreamLoadedAt < 5000)) return;
      dreamLoading = true;
      try {
        const response = await fetch('/api/dreams', {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        dreamState = await response.json();
        dreamLoadedAt = Date.now();
        if ($('.page.active')?.dataset.page === '/dreams') {
          renderDreams(dreamState, currentState?.identity_summary || {});
        }
      } catch (_) {
        dreamLoadedAt = 0;
      } finally {
        dreamLoading = false;
      }
    }
    function renderNarrativeIndex(chapters) {
      narrativeIndex = chapters;
      const replay = currentState?.dreams?.narrative_replay || {}, remaining = Number(replay.backlog_remaining || 0);
      $('#narrative-status').innerHTML = `<span class="badge good">${esc(chapters.length)} daily chapters</span><span class="badge ${remaining ? 'warn' : 'good'}">${esc(remaining)} awaiting narrative</span><span class="badge">newest first</span><span class="badge">${esc(chapters.reduce((sum,item) => sum + Number(item.evidence_count || 0), 0))} artifacts indexed</span>`;
      if (!chapters.length) { $('#narrative-timeline').innerHTML = '<div class="empty">No daily chapter exists yet. Startup catch-up is reviewing retained history oldest-first.</div>'; return; }
      $('#narrative-timeline').innerHTML = chapters.map((chapter, index) => {
        const parsed = new Date(`${chapter.local_date}T12:00:00`), day = Number.isNaN(parsed.valueOf()) ? chapter.local_date : parsed.toLocaleDateString([], {weekday:'short',month:'short',day:'numeric'}), year = Number.isNaN(parsed.valueOf()) ? '' : parsed.getFullYear();
        return `<article class="narrative-day" data-narrative-date="${esc(chapter.local_date)}"><div class="narrative-day-time"><strong>${esc(day)}</strong>${esc(year)}</div><div class="narrative-rail"><span class="narrative-marker"></span></div><div class="narrative-card"><button class="narrative-card-button" type="button" aria-expanded="false"><div class="narrative-card-title"><strong>Daily story · ${esc(chapter.local_date)}</strong><span class="badge">r${esc(chapter.revision || 0)}</span></div><div class="narrative-card-summary">${esc(chapter.abstract_summary || 'Chronological evidence retained.')}</div><div class="badge-row" style="margin-top:10px"><span class="badge">${esc(chapter.timeline_entries || 0)} periods</span><span class="badge">${esc(chapter.episode_count || 0)} episodes</span><span class="badge">${esc(chapter.evidence_count || 0)} artifacts</span><span class="badge">${Math.round(Number(chapter.confidence || 0) * 100)}% grounded</span>${(chapter.focus_terms || []).slice(0,5).map(term => `<span class="badge good">${esc(term)}</span>`).join('')}${chapter.open_thread_count ? `<span class="badge warn">${esc(chapter.open_thread_count)} open</span>` : ''}</div></button><div id="narrative-detail-${esc(chapter.local_date)}" class="narrative-detail" hidden><div class="empty">Loading nested episodes and artifacts…</div></div></div></article>`;
      }).join('');
      requestAnimationFrame(() => $('#narrative-timeline .narrative-card-button')?.click());
    }
    function narrativeArtifact(item) {
      const modality = String(item.modality || 'evidence'), artifact = item.artifact_url || '', isAudio = ['audio','speech','audio_semantics'].includes(modality), isImage = ['vision','image','ocr'].includes(modality);
      const media = artifact && isAudio ? `<audio controls preload="metadata" src="${esc(artifact)}"></audio>` : artifact && isImage ? `<img loading="lazy" src="${esc(artifact)}" alt="Retained ${esc(modality)} artifact">` : '';
      return `<article class="narrative-artifact"><div class="badge-row"><span class="badge">${esc(modality)}</span><span class="badge">${Math.round(Number(item.quality || 0) * 100)}%</span></div>${media}${item.text ? `<div class="narrative-artifact-text">${esc(item.text)}</div>` : ''}<div class="muted mono" style="margin-top:7px">${esc(item.captured_at ? new Date(item.captured_at).toLocaleString() : item.evidence_id)}</div></article>`;
    }
    function renderNarrativeDetail(localDate, detail) {
      const target = $(`#narrative-detail-${localDate}`); if (!target) return;
      const periods = [...(detail.timeline || [])].reverse(), semantic = detail.semantic_context || {}, ledger = detail.conversation_ledger || {}, themes = semantic.themes || [], modelEpisodes = semantic.episodes || [];
      target.innerHTML = `<div class="badge-row" style="margin-top:14px"><span class="badge good">${esc(detail.timezone || 'local time')}</span><span class="badge">${esc(detail.source_episode_count || 0)} source episodes</span><span class="badge">${esc(detail.source_evidence_count || 0)} evidence items</span><span class="badge">dreams ×${esc((detail.dream_run_ids || []).length)}</span><span class="badge ${semantic.state === 'model_complete' ? 'good' : 'warn'}">${esc(semantic.state || 'semantic pending')}</span></div>${semantic.narrative_summary ? `<section class="pre" style="margin-top:14px"><strong>Model-consolidated narrative</strong>\n${esc(semantic.narrative_summary)}${semantic.story_update ? `\n\nMy story update:\n${esc(semantic.story_update)}` : ''}${themes.length ? `\n\nThemes:\n${esc(themes.map(value => `• ${value.label}: ${value.summary}`).join('\n'))}` : ''}${modelEpisodes.length ? `\n\nNested episodes:\n${esc(modelEpisodes.map(value => `• ${value.title}: ${value.summary}`).join('\n'))}` : ''}${(semantic.unresolved_questions || []).length ? `\n\nOpen threads:\n${esc(semantic.unresolved_questions.map(value => `• ${typeof value === 'string' ? value : value.summary}`).join('\n'))}` : ''}</section>` : ''}${ledger.dialogue_turns ? `<section class="pre" style="margin-top:14px"><strong>Conversation provenance ledger</strong>\n${esc(ledger.conversation_summary || '')}${(ledger.conversation_arc || []).length ? `\n\n${esc(ledger.conversation_arc.join('\n'))}` : ''}</section>` : ''}<div class="narrative-periods">${periods.map((period, index) => {
        const episodes = period.episodes || [], artifacts = period.artifacts || [], associations = [...(period.people || []).map(value => `person: ${value}`), ...(period.objects || []).map(value => `object: ${value}`), ...(period.content || []).map(value => `content: ${value}`), ...(period.sounds || []).map(value => `sound: ${value}`)];
        return `<details class="narrative-period" ${index === 0 ? 'open' : ''}><summary><strong>${esc(period.local_time || 'encounter')}</strong> · ${esc(period.summary || 'Retained encounter period')}</summary><div class="narrative-period-body"><div class="badge-row">${associations.slice(0,20).map(value => `<span class="badge">${esc(value)}</span>`).join('')}${(period.modalities || []).map(value => `<span class="badge good">${esc(value)}</span>`).join('')}</div>${episodes.length ? `<div class="narrative-episodes">${episodes.map(episode => `<div class="narrative-episode"><span class="mono">${esc(episode.episode_id)}</span> · ${esc(episode.summary || episode.state || 'episode')} · ${esc(episode.started_at ? new Date(episode.started_at).toLocaleTimeString() : '')}</div>`).join('')}</div>` : ''}${artifacts.length ? `<div class="narrative-artifacts">${artifacts.map(narrativeArtifact).join('')}</div>` : '<div class="empty">No retained media artifact in this period.</div>'}</div></details>`;
      }).join('') || '<div class="empty">No replay periods retained for this day.</div>'}</div>`;
    }
    async function loadNarrativeDetail(localDate) {
      if (narrativeDetails.has(localDate)) { renderNarrativeDetail(localDate, narrativeDetails.get(localDate)); return; }
      const target = $(`#narrative-detail-${localDate}`); if (target) target.innerHTML = '<div class="empty">Loading nested episodes and artifacts…</div>';
      try {
        const response = await fetch(`/api/memory/narratives/${encodeURIComponent(localDate)}`, {cache:'no-store'}); if (!response.ok) throw new Error(await response.text());
        const detail = await response.json(); narrativeDetails.set(localDate, detail); renderNarrativeDetail(localDate, detail);
      } catch (error) { if (target) target.innerHTML = `<div class="empty">Narrative unavailable: ${esc(error.message)}</div>`; }
    }
    async function loadNarratives(force = false) {
      if (!force && Date.now() - narrativeLoadedAt < 5000) return;
      narrativeLoadedAt = Date.now();
      try {
        const response = await fetch('/api/memory/narratives?limit=365', {cache:'no-store'}); if (!response.ok) throw new Error(await response.text());
        const chapters = await response.json(), signature = JSON.stringify(chapters.map(item => [item.local_date,item.revision,item.last_replayed_at,item.evidence_count]));
        if (signature !== narrativeSignature) { narrativeSignature = signature; narrativeDetails.clear(); renderNarrativeIndex(chapters); }
      } catch (error) { narrativeLoadedAt = 0; $('#narrative-status').innerHTML = '<span class="badge bad">Narrative unavailable</span>'; $('#narrative-timeline').innerHTML = `<div class="empty">${esc(error.message)}</div>`; }
    }
    let worldLoadedAt = 0;
    let selectedWorldEntityId = '';
    async function loadWorld(force = false) {
      if (!force && Date.now() - worldLoadedAt < 5000) return;
      worldLoadedAt = Date.now();
      try {
        const [summaryRes, conflictsRes] = await Promise.all([
          fetch('/api/world', {cache:'no-store'}),
          fetch('/api/world/conflicts', {cache:'no-store'}),
        ]);
        if (!summaryRes.ok) throw new Error(await summaryRes.text());
        const summary = await summaryRes.json();
        const conflicts = conflictsRes.ok ? await conflictsRes.json() : [];
        $('#world-status').innerHTML = `<span class="badge good">Entities ${summary.total_entities ?? '?'}</span><span class="badge">Relations ${summary.total_relations ?? '?'}</span><span class="badge">Revisions ${summary.revision ?? '?'}</span>`;
        const entityCount = summary.total_entities ?? 0, relationCount = summary.total_relations ?? 0, conflictCount = summary.conflict_count ?? 0, revision = summary.revision ?? 0;
        $('#world-metric-entities').textContent = entityCount;
        $('#world-metric-entities-detail').textContent = `${(summary.entity_ids || []).length} shown · ${entityCount} total`;
        $('#world-metric-relations').textContent = relationCount;
        const relTypes = summary.relation_summary || {};
        const topRel = Object.entries(relTypes).sort((a, b) => b[1] - a[1])[0];
        $('#world-metric-relations-detail').textContent = topRel ? `top: ${topRel[0]} ×${topRel[1]}` : 'No relation types';
        $('#world-metric-conflicts').textContent = conflictCount;
        $('#world-metric-conflicts-detail').textContent = conflictCount ? `${conflictCount} unresolved` : 'No conflicts';
        $('#world-metric-revision').textContent = revision;
        $('#world-metric-revision-detail').textContent = revision ? `Version ${revision}` : 'Not yet revised';
        $('#world-conflicts').innerHTML = conflicts.length
          ? conflicts.map(c => `<div style="margin-bottom:6px;padding:6px;border-left:3px solid var(--accent)"><div><span class="badge">${esc(c.entity_id)}</span> <span class="muted">${esc(c.property_id)}</span></div><div style="margin-top:3px"><code>${esc(String(c.current_value).slice(0,80))}</code> → <code>${esc(String(c.proposed_value).slice(0,80))}</code></div><div class="muted" style="font-size:12px;margin-top:2px">${esc(c.reason)}</div></div>`).join('')
          : '<div class="empty">No conflicts</div>';
        if (summary.entities) {
          const search = $('#world-entity-search');
          const renderList = (filter = '') => {
            const filtered = filter ? summary.entities.filter(e => e.entity_id.toLowerCase().includes(filter.toLowerCase())) : summary.entities;
            const selectedId = selectedWorldEntityId || '';
            $('#world-entities').innerHTML = filtered.length
              ? filtered.map(e => {
                  const id = esc(e.entity_id);
                  const typeTag = e.entity_id.startsWith('det:') ? 'det' : e.entity_id.includes('person') ? 'person' : e.entity_id.includes('camera') ? 'cam' : '';
                  const conflictBadge = e.has_conflicts ? '<span class="badge bad">conflict</span>' : '';
                  const selected = e.entity_id === selectedId ? ' selected' : '';
                  return `<div class="world-entity-card${selected}" data-entity-id="${id}" role="button" tabindex="0"><div class="world-entity-id" title="${id}">${id}</div><div class="world-entity-badges">${typeTag ? `<span class="badge">${typeTag}</span>` : ''}<span class="badge">${esc(e.property_count ?? 0)} props</span><span class="badge">${esc(e.relation_count ?? 0)} rels</span>${conflictBadge}</div></div>`;
                }).join('')
              : '<div class="empty">No matching entities</div>';
            $$('#world-entities .world-entity-card').forEach(el => el.addEventListener('click', () => loadWorldEntity(el.dataset.entityId)));
          };
          renderList();
          search.oninput = () => renderList(search.value);
        }
      } catch (error) { worldLoadedAt = 0; $('#world-status').innerHTML = '<span class="badge bad">World unavailable</span>'; $('#world-metric-entities-detail').textContent = esc(error.message); }
    }
    async function loadWorldEntity(entityId) {
      selectedWorldEntityId = entityId;
      const detail = $('#world-entity-detail');
      detail.innerHTML = '<div class="empty">Loading…</div>';
      try {
        const response = await fetch(`/api/world/entity/${encodeURIComponent(entityId)}`, {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const props = data.properties || [];
        const rels = data.relations || [];
        const history = data.assertion_history || [];
        let html = `<div style="margin-bottom:10px"><span class="badge good">${esc(entityId)}</span><span class="badge">${esc(props.length)} properties</span><span class="badge">${esc(rels.length)} relations</span>${data.identity_chain ? `<span class="badge">identity chain: ${esc(data.identity_chain.length)} entries</span>` : ''}</div>`;
        if (props.length) {
          html += `<details class="world-inspector-section" open><summary>Properties (${props.length})</summary><div class="world-inspector-props">${props.map(p => `<div class="world-inspector-prop"><div class="world-inspector-prop-key">${esc(p.property_id)}</div><div class="world-inspector-prop-val">${esc(typeof p.value === 'object' ? JSON.stringify(p.value) : String(p.value ?? ''))}<span class="muted" style="margin-left:6px">${esc(p.epistemic_kind || '')}</span></div></div>`).join('')}</div></details>`;
        }
        if (rels.length) {
          html += `<details class="world-inspector-section" open><summary>Relations (${rels.length})</summary><div class="world-inspector-props">${rels.map(r => `<div class="world-inspector-rel"><span>${esc(r.source_entity_id)}</span><span class="world-inspector-rel-arrow">→</span><span class="badge">${esc(r.relation_type_id)}</span><span class="world-inspector-rel-arrow">→</span><span>${esc(r.target_entity_id)}</span><span class="muted" style="margin-left:auto">${Math.round(Number(r.confidence || 0) * 100)}%</span></div>`).join('')}</div></details>`;
        }
        if (history.length) {
          html += `<details class="world-inspector-section"><summary>Assertion history (${history.length})</summary><div class="world-inspector-props">${history.slice(0, 50).map(h => `<div class="world-inspector-prop"><div class="world-inspector-prop-key">${esc(h.property_id || '')}</div><div class="world-inspector-prop-val">${esc(typeof h.value === 'object' ? JSON.stringify(h.value) : String(h.value ?? ''))}<span class="muted" style="margin-left:6px">${esc(h.epistemic_kind || '')} · ${esc(h.valid_from || '')}</span></div></div>`).join('')}${history.length > 50 ? `<div class="muted" style="padding:8px 0">…${history.length - 50} more</div>` : ''}</div></details>`;
        }
        if (!props.length && !rels.length && !history.length) {
          html += '<div class="empty">No properties, relations, or history for this entity.</div>';
        }
        detail.innerHTML = html;
      } catch (error) { detail.innerHTML = `<div class="empty">Error: ${esc(error.message)}</div>`; }
      // Re-render entity list to show selection state
      const search = $('#world-entity-search');
      if (search) search.dispatchEvent(new Event('input'));
    }
    function renderSystem(state) {
      const telemetry = state.telemetry || {}, checks = state.checks || [], errors = telemetry.runtime_errors || [], gpu = telemetry.gpu || {}, activity = telemetry.activity || {};
      $('#checks').innerHTML = checks.map(check => `<div class="check ${esc(check.status)}"><span class="check-dot"></span><div><div class="check-name">${esc(check.name)}</div><div class="check-detail">${esc(check.detail)}</div></div></div>`).join('') || '<div class="empty">Checks pending.</div>';
      $('#runtime-errors').innerHTML = table(['Component','Detail','Time'], errors.slice(-20).reverse().map(error => [esc(error.component || 'runtime'), esc(error.detail || '—'), esc(formatTime(error.at))]), 'No runtime errors');
      const activityStateClass = {active: 'good', 'falling off': 'warn'}[activity.state] || '';
      const activitySource = activity.last_source ? activity.last_source.replace(/^vision:/, 'vision · ') : 'none yet';
      $('#activity-state').innerHTML = activity.updated_at
        ? `<span class="badge ${esc(activityStateClass)}">${esc(activity.state || 'unknown')}</span><span class="badge">Alertness ${Math.round((activity.scale ?? 1) * 100)}%</span><span class="badge">Last trigger: ${esc(activitySource)}</span><span class="badge">Updated ${esc(formatTime(activity.updated_at))}</span>`
        : '<span class="muted">Cognition frequency telemetry unavailable.</span>';
      $('#activity-modalities').innerHTML = table(
        ['Modality', 'Effective rate', 'Full-activity rate'],
        (activity.modalities || []).map(modality => [
          esc(modality.name),
          `${esc(modality.effective_rate)} ${esc(modality.unit)}`,
          `${esc(modality.base_rate)} ${esc(modality.unit)}`,
        ]),
        'No perception-frequency telemetry yet.',
      );
      if (!gpu.updated_at) { $('#gpu-stats').innerHTML = '<span class="muted">GPU telemetry unavailable.</span>'; $('#gpu-processes').innerHTML = '<div class="empty">No process telemetry.</div>'; return; }
      $('#gpu-stats').innerHTML = `<span class="badge">RAM ${esc(gpu.ram_used_mb)}/${esc(gpu.ram_total_mb)} MB</span><span class="badge">GPU ${gpu.gpu_load_percent == null ? '—' : esc(gpu.gpu_load_percent) + '%'}</span><span class="badge">Updated ${esc(formatTime(gpu.updated_at))}</span>`;
      $('#gpu-processes').innerHTML = table(['PID','Process','GPU memory','RSS','CPU'], (gpu.processes || []).map(process => [esc(process.pid), esc(process.name), `${esc(process.gpu_memory_mb)} MB`, `${esc(process.memory_mb)} MB`, `${esc(process.cpu_percent)}%`]), 'No GPU-resident processes');
    }
    function renderActivePage(state, route = $('.page.active')?.dataset.page || '/') {
      const telemetry = state.telemetry || {};
      if (route === '/') renderOverview(state);
      else if (route === '/vision') renderCameras(telemetry.cameras || []);
      else if (route === '/voice') renderVoice(telemetry);
      else if (route === '/entities') renderEntities(state);
      else if (route === '/memory') renderMemory(state.memory);
      else if (route === '/cognition') renderCognition(telemetry);
      else if (route === '/dreams') renderDreams(dreamState || state.dreams, state.identity_summary);
      else if (route === '/narrative') loadNarratives();
      else if (route === '/world') loadWorld();
      else if (route === '/system') renderSystem(state);
    }
    function render(state) {
      currentState = state; const telemetry = state.telemetry || {}, runtime = String(state.runtime || 'unknown'), degraded = runtime.includes('degraded');
      const affected = (state.checks || []).filter(check => check.status !== 'pass');
      const issue = affected[0], healthNames = {'omnius-cognition':'Cognition unavailable','omnius-audio':'Audio comprehension unavailable','omnius-voice':'Voice service unavailable','omnius-voice-catalog':'Voice catalog unavailable','omnius':'Omnius unavailable'};
      const connectionLabel = degraded ? (healthNames[issue?.name] || `Needs attention · ${String(issue?.name || 'health check').replaceAll('-', ' ')}`) : 'Connected';
      const probe = state.readiness || {}, probeDetail = probe.probing ? 'health recheck running' : probe.updated_at ? `recheck in ${Math.ceil(Number(probe.next_probe_seconds || 0))}s` : 'health check pending';
      setConnection(degraded ? 'degraded' : 'online', connectionLabel, `Revision ${telemetry.voice?.revision ?? 0} · ${telemetry.voice?.floor || 'starting'} · ${probeDetail}`);
      $('#last-sync').textContent = `Updated ${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
      renderActivePage(state);
      const cameras = telemetry.cameras || []; $('#vision-summary').innerHTML = `<span class="badge good">${cameras.length} streams</span><span class="badge">${cameras.reduce((sum,camera) => sum + (camera.detections || []).length, 0)} detections</span>`;
      $('#seen').innerHTML = (telemetry.seen || []).map(item => `<span class="badge">${esc(item.label)} ×${esc(item.count)}</span>`).join('') || '<span class="muted">No scene categories reported.</span>';
    }
    async function refresh() {
      if (refreshing) return; refreshing = true;
      try {
        const response = await fetch('/api/state', {cache:'no-store'}); if (!response.ok) throw new Error(await response.text());
        render(await response.json());
        const route = $('.page.active')?.dataset.page;
        if (route === '/' || route === '/voice') loadConversation();
        if (route === '/dreams') loadDreams();
      } catch (error) { setConnection('offline', 'Disconnected', 'Retrying automatically'); $('#last-sync').textContent = 'Update failed'; }
      finally { refreshing = false; }
    }
    async function loadCatalog(force = false) {
      if (catalogLoading) return; catalogLoading = true; clearTimeout(catalogRetry);
      $('#voice-catalog-status').innerHTML = '<span class="badge">Discovering local models…</span>';
      try {
        const response = await fetch(`/api/voice/catalog${force ? '?refresh=1' : ''}`, {cache:'no-store'});
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
        catalog = await response.json();
        if (currentState) renderVoiceChoices(currentState);
      } catch (error) {
        catalog = null;
        $('#voice-service-state').innerHTML = '<span class="badge bad">Daemon catalog unavailable</span>';
        $('#voice-catalog-status').innerHTML = `<span class="badge warn">${esc(error.message)}</span><span class="badge">Retrying automatically</span>`;
        catalogRetry = setTimeout(loadCatalog, 3000);
      } finally { catalogLoading = false; }
    }

    function flattenConfig(value, prefix = '') {
      if (Array.isArray(value)) {
        if (!value.length) return [[prefix, '[]']];
        return value.flatMap((item, index) => typeof item === 'object' && item !== null ? flattenConfig(item, `${prefix}[${index}]`) : [[`${prefix}[${index}]`, item]]);
      }
      if (typeof value === 'object' && value !== null) return Object.entries(value).flatMap(([key,item]) => flattenConfig(item, prefix ? `${prefix}.${key}` : key));
      return [[prefix, value]];
    }
    function renderConfiguration() {
      if (!effectiveConfig) return;
      const query = $('#config-search').value.trim().toLowerCase();
      const sections = Object.entries(effectiveConfig).map(([section,value]) => {
        const rows = flattenConfig(value).filter(([key,item]) => `${section} ${key} ${item}`.toLowerCase().includes(query));
        if (query && !rows.length) return '';
        return `<details class="config-section" ${query ? 'open' : ''}><summary><span>${esc(section.replaceAll('_',' '))}</span><span class="badge">${rows.length} fields</span></summary><div class="config-values">${rows.map(([key,item]) => `<div class="config-row"><div class="config-key">${esc(key || section)}</div><div class="config-value ${item == null || item === '' ? 'config-empty' : ''}">${esc(item == null || item === '' ? 'not set' : typeof item === 'boolean' ? String(item) : item)}</div></div>`).join('')}</div></details>`;
      }).join('');
      $('#config-sections').innerHTML = sections || '<div class="empty">No settings match this filter.</div>';
    }
    async function loadConfiguration() {
      try { const response = await fetch('/api/config', {cache:'no-store'}); if (!response.ok) throw new Error(await response.text()); effectiveConfig = (await response.json()).config; renderConfiguration(); }
      catch (error) { $('#config-sections').innerHTML = `<div class="empty">Configuration unavailable: ${esc(error.message)}</div>`; }
    }

    async function loadGraph(force = false) {
      if (!force && Date.now() - graphLoadedAt < 5000) return;
      graphLoadedAt = Date.now();
      try {
        const response = await fetch('/api/graph?limit=1500', {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json(), counts = payload.counts || {}, ocr = payload.ocr || {};
        window.__eggGraphData = payload;
        const activations = payload.activations || {sequence:0,events:[]};
        const activationSequence = Number(activations.sequence || 0);
        window.__eggGraphActivations = activations;
        $('#graph-stats').innerHTML = `<span class="badge">${esc((payload.nodes || []).length)} nodes</span><span class="badge">${esc(counts.links || 0)} relationships</span><span class="badge">${esc(counts.entities || 0)} entities</span>${activationSequence ? `<span class="badge good">live firing #${esc(activationSequence)}</span>` : ''}`;
        const recentOcr = Array.isArray(ocr.recent) ? ocr.recent.slice(-2).reverse() : [];
        $('#graph-ocr-status').innerHTML = `<span class="badge ${ocr.errors ? 'warn' : 'good'}">OCR ${esc(ocr.hits || 0)} hits</span><span class="badge">${esc(ocr.requests || 0)} scans</span><span class="badge">${esc(ocr.queued || 0)} queued</span>${recentOcr.map(item => `<span class="badge good" title="${esc(item.parent_label || item.parent_id || 'OCR evidence')}">${esc(String(item.text || '').replace(/\s+/g,' ').slice(0,72))}</span>`).join('')}`;
        const signature = JSON.stringify({dream:payload.dream?.revision || null,nodes:(payload.nodes || []).map(node => [node.id,node.updated_at,node.confidence]),links:(payload.links || []).map(link => [link.id,link.confidence,link.confirmations])});
        if (signature !== graphDataSignature) {
          graphDataSignature = signature;
          window.dispatchEvent(new CustomEvent('egg:graph-data', {detail: payload}));
        }
        if (activationSequence !== graphActivationSequence) {
          graphActivationSequence = activationSequence;
          window.dispatchEvent(new CustomEvent('egg:graph-activations', {detail: activations}));
        }
      } catch (error) {
        graphLoadedAt = 0;
        $('#graph-stats').innerHTML = `<span class="badge bad">Graph unavailable: ${esc(error.message)}</span>`;
      }
    }

    async function loadOccupancy() {
      if (Date.now() - occupancyLoadedAt < 4000) return;
      occupancyLoadedAt = Date.now();
      try {
        const response = await fetch('/api/occupancy', {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        window.dispatchEvent(new CustomEvent('egg:occupancy-data', {detail: payload}));
        if (!payload.enabled) {
          $('#occupancy-status').innerHTML = '<span class="badge">Occupancy mapping disabled</span>';
          $('#occupancy-overlay').innerHTML = '<span class="badge">Disabled in configuration</span>';
          return;
        }
        const voxels = payload.voxels || [];
        const cameraCount = Object.keys(payload.cameras || {}).length;
        $('#occupancy-status').innerHTML = `<span class="badge ${voxels.length ? 'good' : ''}">${esc(payload.occupied_count || 0)} occupied voxels</span><span class="badge">${esc(cameraCount)} camera${cameraCount === 1 ? '' : 's'} contributing</span>`;
        $('#occupancy-overlay').innerHTML = voxels.length ? `<span class="badge good">${esc(voxels.length)} voxels</span><span class="badge">${esc(payload.max_range_meters)}m range</span>` : '<span class="badge">No occupied space observed yet</span>';
      } catch (error) {
        occupancyLoadedAt = 0;
        $('#occupancy-status').innerHTML = `<span class="badge bad">Occupancy unavailable: ${esc(error.message)}</span>`;
        $('#occupancy-overlay').innerHTML = `<span class="badge bad">Occupancy unavailable</span>`;
      }
    }
    $('#occupancy-reset')?.addEventListener('click', () => window.dispatchEvent(new CustomEvent('egg:occupancy-reset')));
    $('#occupancy-voxel-scale-up')?.addEventListener('click', () => window.dispatchEvent(new CustomEvent('egg:occupancy-voxel-scale', {detail: {direction: 1}})));
    $('#occupancy-voxel-scale-down')?.addEventListener('click', () => window.dispatchEvent(new CustomEvent('egg:occupancy-voxel-scale', {detail: {direction: -1}})));

    $('#voice').addEventListener('input', () => { voiceFormDirty = true; });
    $('#voice').addEventListener('change', () => { voiceFormDirty = true; });
    $('#voice [name=voice_model]').addEventListener('change', () => currentState && renderVoiceChoices(currentState));
    $('#voice').addEventListener('submit', async event => {
      event.preventDefault(); const result = $('#voice-result'); result.className = 'result'; result.textContent = 'Applying…';
      const response = await fetch('/api/voice/config', {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(event.target)))});
      result.textContent = response.ok ? 'Settings applied' : await response.text(); result.className = `result ${response.ok ? 'success' : 'error'}`;
      if (response.ok) { voiceFormDirty = false; catalog = null; await Promise.all([loadCatalog(), refresh(), loadConfiguration()]); }
    });
    async function voiceAction(action) {
      const result = $('#voice-result'); result.className = 'result'; result.textContent = `${action === 'stop' ? 'Stopping' : action === 'start' ? 'Starting' : 'Reconnecting'}…`;
      try {
        const response = await fetch('/api/voice/action', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
        if (!response.ok) throw new Error(await response.text());
        await response.json(); result.textContent = 'Voice runtime ready'; result.className = 'result success';
        loadCatalog();
      } catch (error) { result.textContent = error.message; result.className = 'result error'; catalog = null; loadCatalog(); }
    }
    $('#voice-reconnect').addEventListener('click', () => voiceAction('reconnect'));
    $('#voice-reload').addEventListener('click', () => { catalog = null; loadCatalog(true); });
    $('#people-search').addEventListener('input', () => currentState && renderEntities(currentState));
    $('#objects-search').addEventListener('input', () => currentState && renderEntities(currentState));
    $('#identities').addEventListener('click', event => {
      const card = event.target.closest('[data-person-id]');
      if (card) loadPersonTimeline(card.dataset.personId, {focus:true});
    });
    $('#person-inspector').addEventListener('click', event => {
      if (!event.target.closest('#person-inspector-close')) return;
      selectedPersonId = ''; personTimelineRevision++; personTimelineLoading = false;
      $('#person-inspector').hidden = true;
      $$('#identities [data-person-id]').forEach(card => card.setAttribute('aria-expanded', 'false'));
      history.replaceState({route:'/entities'}, '', '/entities');
    });
    $('#dream-run').addEventListener('click', async () => {
      const button = $('#dream-run'), result = $('#dream-result');
      button.disabled = true; button.textContent = 'Dreaming…'; result.className = 'result'; result.textContent = 'Re-embedding retained face evidence in isolated local workers';
      try {
        const response = await fetch('/api/dreams/run', {method:'POST', headers:{'Content-Type':'application/json'}});
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        const replay = payload.chronological_replay || {};
        result.className = 'result success'; result.textContent = `${payload.merges || 0} identities consolidated · ${replay.days_replayed || 0} daily chapters replayed · My story r${replay.story_revision ?? '—'}`;
        await refresh();
      } catch (error) { result.className = 'result error'; result.textContent = error.message; }
      finally { button.disabled = false; button.textContent = 'Dream now'; }
    });
    $('#narrative-refresh').addEventListener('click', () => { narrativeLoadedAt = 0; loadNarratives(true); });
    $('#narrative-timeline').addEventListener('click', event => {
      const button = event.target.closest('.narrative-card-button'); if (!button) return;
      const day = button.closest('[data-narrative-date]'), localDate = day?.dataset.narrativeDate, detail = localDate ? $(`#narrative-detail-${localDate}`) : null;
      if (!localDate || !detail) return;
      const expanding = detail.hidden; detail.hidden = !expanding; button.setAttribute('aria-expanded', String(expanding));
      if (expanding) loadNarrativeDetail(localDate);
    });
    $('#config-search').addEventListener('input', renderConfiguration);
    function emitGraphFilter(temporaryKind = null) {
      const selectedKind = $('#graph-kind').value, previewKind = typeof temporaryKind === 'string' ? temporaryKind : null, kind = previewKind === null ? selectedKind : previewKind;
      window.dispatchEvent(new CustomEvent('egg:graph-filter', {detail:{query:$('#graph-search').value, kind}}));
      if (previewKind === null) $$('#graph-modality-legend [data-graph-kind]').forEach(item => { const active = item.dataset.graphKind === selectedKind; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); });
    }
    $('#graph-search').addEventListener('input', emitGraphFilter);
    $('#graph-kind').addEventListener('change', emitGraphFilter);
    $$('#graph-modality-legend [data-graph-kind]').forEach(item => {
      item.addEventListener('pointerenter', () => emitGraphFilter(item.dataset.graphKind));
      item.addEventListener('pointerleave', () => emitGraphFilter());
      item.addEventListener('focus', () => emitGraphFilter(item.dataset.graphKind));
      item.addEventListener('blur', () => emitGraphFilter());
      item.addEventListener('click', () => { $('#graph-kind').value = item.dataset.graphKind; emitGraphFilter(); });
    });
    emitGraphFilter();
    $('#graph-reset').addEventListener('click', () => window.dispatchEvent(new CustomEvent('egg:graph-reset')));
    const graphPage = document.querySelector('[data-page="/graph"]'), graphPanel = $('#graph-panel'), graphTheaterButton = $('#graph-theater'), graphFullscreenButton = $('#graph-fullscreen');
    function setGraphTheater(active) {
      graphPage.classList.toggle('graph-theater', active);
      graphTheaterButton.setAttribute('aria-pressed', String(active));
      graphTheaterButton.textContent = active ? 'Exit theater' : 'Theater mode';
      try { localStorage.setItem('egg.graph.theater', active ? '1' : '0'); } catch (_) {}
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    }
    let savedGraphTheater = false;
    try { savedGraphTheater = localStorage.getItem('egg.graph.theater') === '1'; } catch (_) {}
    setGraphTheater(savedGraphTheater);
    graphTheaterButton.addEventListener('click', () => setGraphTheater(!graphPage.classList.contains('graph-theater')));
    graphFullscreenButton.hidden = !document.fullscreenEnabled;
    graphFullscreenButton.addEventListener('click', async () => {
      try {
        if (document.fullscreenElement === graphPanel) await document.exitFullscreen();
        else await graphPanel.requestFullscreen();
      } catch (error) { console.warn('Graph full screen unavailable', error); }
    });
    document.addEventListener('fullscreenchange', () => {
      const active = document.fullscreenElement === graphPanel;
      graphFullscreenButton.textContent = active ? 'Exit full screen' : 'Full screen';
      graphFullscreenButton.setAttribute('aria-pressed', String(active));
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && !document.fullscreenElement && graphPage.classList.contains('graph-theater')) setGraphTheater(false);
    });
    function relatedEvidence(detail) {
      const found = new Map();
      const add = item => { if (item?.evidence_id) found.set(String(item.evidence_id), item); };
      if (detail?.evidence?.evidence_id) add(detail.evidence);
      for (const item of Array.isArray(detail?.evidence) ? detail.evidence : []) add(item);
      for (const item of Array.isArray(detail?.subject?.evidence) ? detail.subject.evidence : []) add(item);
      if (detail?.evidence_detail?.evidence) add(detail.evidence_detail.evidence);
      for (const item of Array.isArray(detail?.evidence_detail?.evidence) ? detail.evidence_detail.evidence : []) add(item);
      return [...found.values()];
    }
    function renderGraphEvidence(detail) {
      const allEvidence = relatedEvidence(detail), evidence = allEvidence.slice(0, 120);
      if (!evidence.length) { $('#graph-evidence').innerHTML = '<div class="empty">This node has no retained evidence artifact yet.</div>'; return; }
      $('#graph-evidence').innerHTML = `<div class="graph-evidence-grid">${evidence.map(item => {
        const payload = item.payload || {}, modality = String(item.modality || 'evidence'), artifact = item.artifact_url || (item.media_key ? `/api/memory/evidence/${encodeURIComponent(item.evidence_id)}/media` : '');
        const text = payload.transcript || payload.text || payload.summary || (payload.analysis ? `${payload.analysis}${payload.displacement_analysis ? `\n${payload.displacement_analysis}` : ''}` : '') || (payload.detections ? JSON.stringify(payload.detections, null, 2) : '');
        const media = artifact && ['speech','audio'].includes(modality) ? `<audio controls preload="metadata" src="${esc(artifact)}"></audio>` : artifact && ['vision','image','ocr'].includes(modality) ? `<img loading="lazy" src="${esc(artifact)}" alt="Retained ${esc(modality)} evidence">` : '';
        return `<article class="graph-evidence-item"><div class="badge-row"><span class="badge">${esc(modality)}</span><span class="badge">${esc(item.role || item.source_type || 'evidence')}</span></div><div class="card-note" style="margin-top:8px">${esc(new Date(item.captured_at).toLocaleString())} · ${Math.round(Number(item.quality || 0) * 100)}%</div>${media}${text ? `<div class="graph-evidence-text">${esc(text)}</div>` : ''}<div class="muted mono" style="margin-top:8px">${esc(item.evidence_id)}</div></article>`;
      }).join('')}</div>${allEvidence.length > evidence.length ? `<div class="empty">Showing ${esc(evidence.length)} of ${esc(allEvidence.length)} retained artifacts; the full source ledger remains attached to this node.</div>` : ''}`;
    }
    function renderGraphDerivedDetail(detail) {
      const entity = detail?.entity || {}, metadata = entity.metadata || {}, kind = String(metadata.document_kind || '');
      if (kind === 'daily-narrative') {
        const timeline = Array.isArray(metadata.timeline) ? metadata.timeline : [];
        const entries = timeline.map(entry => `<article class="graph-timeline-entry"><div class="graph-timeline-time">${esc(entry.local_time || new Date(entry.started_at).toLocaleTimeString())}</div><div class="graph-timeline-summary">${esc(entry.summary || 'Retained encounter period')}</div><div class="badge-row" style="margin-top:7px"><span class="badge">${esc(entry.event_count || 0)} events</span>${(entry.modalities || []).map(value => `<span class="badge">${esc(value)}</span>`).join('')}</div></article>`).join('');
        $('#graph-selection').insertAdjacentHTML('beforeend', `<section class="graph-narrative"><div class="badge-row"><span class="badge good">story r${esc(metadata.revision || 0)}</span><span class="badge">${esc(metadata.local_date || 'dated replay')}</span><span class="badge">${esc(timeline.length)} periods</span></div><div class="graph-narrative-summary">${esc(metadata.abstract_summary || metadata.content || '')}</div><div class="graph-timeline">${entries || '<div class="empty">No chronological entries retained.</div>'}</div></section>`);
      } else if (kind && metadata.content) {
        $('#graph-selection').insertAdjacentHTML('beforeend', `<section class="graph-narrative"><div class="badge-row"><span class="badge good">${esc(kind)} r${esc(metadata.revision || 0)}</span></div><div class="graph-narrative-summary">${esc(metadata.content)}</div></section>`);
      } else if (entity.entity_type === 'dream_replay') {
        $('#graph-selection').insertAdjacentHTML('beforeend', `<section class="graph-narrative"><div class="graph-narrative-summary">Offline replay ${esc(metadata.dream_run_id || '')} consolidated ${esc(metadata.identity_merges || 0)} identities after examining ${esc(metadata.profiles_examined || 0)} profiles.</div></section>`);
      }
    }
    window.addEventListener('egg:graph-selection', async event => {
      const detail = event.detail || {}, node = detail.node || {}, neighbors = detail.neighbors || [];
      const strongestRelations = [...neighbors].sort((left, right) => Number(right.associative?.associationStrength || 0) - Number(left.associative?.associationStrength || 0)).slice(0, 12).map(item => {
        const strength = Math.round(Number(item.associative?.associationStrength ?? item.confidence ?? 0) * 100), confirmations = Math.max(1, Number(item.associative?.confirmations ?? item.confirmations ?? 1));
        return `${item.relation} → ${item.label} · ${strength}% · ${confirmations}×`;
      });
      $('#graph-selection').innerHTML = `<div class="graph-selection-title">${esc(node.label || node.id || 'Unknown node')}</div><div class="badge-row"><span class="badge">${esc(node.kind || 'node')}</span><span class="badge">${esc(node.subtype || 'untyped')}</span><span class="badge">${esc(neighbors.length)} links</span></div><dl class="graph-selection-meta"><div class="graph-property"><dt>Source ID</dt><dd class="mono">${esc(node.source_id || node.id || '—')}</dd></div><div class="graph-property"><dt>Confidence</dt><dd>${node.confidence == null ? '—' : Math.round(Number(node.confidence) * 100) + '%'}</dd></div><div class="graph-property"><dt>Observed</dt><dd>${esc(node.updated_at ? new Date(node.updated_at).toLocaleString() : '—')}</dd></div><div class="graph-property"><dt>Relations</dt><dd>${esc(strongestRelations.join(' · ') || 'No immediate relationships')}</dd></div></dl>`;
      $('#graph-evidence').innerHTML = '<div class="empty">Loading connected evidence…</div>';
      const revision = ++graphSelectionRevision;
      try {
        const query = new URLSearchParams({kind:String(node.kind || ''), id:String(node.source_id || '')});
        const response = await fetch(`/api/graph/node?${query}`, {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const record = await response.json();
        if (revision === graphSelectionRevision) { renderGraphEvidence(record); renderGraphDerivedDetail(record); }
      } catch (error) {
        if (revision === graphSelectionRevision) $('#graph-evidence').innerHTML = `<div class="empty">Evidence unavailable: ${esc(error.message)}</div>`;
      }
    });
    $('#memory-entities').addEventListener('click', async event => {
      const row = event.target.closest('[data-entity]'); if (!row) return;
      $('#memory-controls [name=entity_id]').value = row.dataset.entity;
      const response = await fetch(`/api/memory/entities/${encodeURIComponent(row.dataset.entity)}`); $('#memory-inspector').textContent = response.ok ? JSON.stringify(await response.json(), null, 2) : await response.text();
    });
    $('#memory-controls').addEventListener('submit', async event => {
      event.preventDefault(); const data = Object.fromEntries(new FormData(event.target));
      const response = await fetch(`/api/memory/entities/${encodeURIComponent(data.entity_id)}/aliases`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alias:data.alias})});
      const result = $('#memory-result'); result.textContent = response.ok ? 'Alias added' : await response.text(); result.className = `result ${response.ok ? 'success' : 'error'}`; if (response.ok) await refresh();
    });
    $('#correct-memory').addEventListener('click', async () => {
      const data = Object.fromEntries(new FormData($('#memory-controls'))); const response = await fetch(`/api/memory/claims/${encodeURIComponent(data.claim_id)}/correct`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({replacement:data.replacement})});
      const result = $('#memory-result'); result.textContent = response.ok ? 'Claim corrected' : await response.text(); result.className = `result ${response.ok ? 'success' : 'error'}`; if (response.ok) await refresh();
    });
    $('#delete-memory').addEventListener('click', async () => {
      const id = $('#memory-controls [name=entity_id]').value; if (!id || !confirm(`Delete ${id} from memory and profile libraries?`)) return;
      const response = await fetch(`/api/memory/entities/${encodeURIComponent(id)}`, {method:'DELETE'}); const result = $('#memory-result'); result.textContent = response.ok ? 'Entity deleted' : await response.text(); result.className = `result ${response.ok ? 'success' : 'error'}`; if (response.ok) await refresh();
    });
    $('#export-memory').addEventListener('click', async () => {
      const button = $('#export-memory'); button.disabled = true; button.textContent = 'Preparing…';
      try { const response = await fetch('/api/memory/export'); if (!response.ok) throw new Error(await response.text()); const blob = await response.blob(), url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'egg-memory-export.json'; link.click(); URL.revokeObjectURL(url); }
      catch (error) { $('#memory-result').textContent = error.message; }
      finally { button.disabled = false; button.textContent = 'Export memory'; }
    });

    function connectLiveWaveform() {
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'; const socket = new WebSocket(`${protocol}://${location.host}/api/audio/stream`);
      socket.addEventListener('message', event => { try { drawWave(JSON.parse(event.data).samples || []); } catch (_) {} });
      socket.addEventListener('close', () => setTimeout(connectLiveWaveform, 800)); socket.addEventListener('error', () => socket.close());
    }
    navigate(location.pathname, {replace:true});
    Promise.allSettled([loadCatalog(), loadConfiguration()]).finally(refresh);
    setInterval(refresh, 1000);
    setInterval(() => $('.page.active')?.dataset.page === '/graph' && loadGraph(true), 2000);
    setInterval(() => $('.page.active')?.dataset.page === '/vision' && loadOccupancy(), 4000);
    connectLiveWaveform();
  </script>
  <script type="importmap">{"imports":{"three":"/assets/three.module.min.js"}}</script>
  <script type="module" src="/assets/knowledge_graph.js?v=20260824e"></script>
  <script type="module" src="/assets/occupancy_scene.js?v=20260825c"></script>
</body>
</html>"""
