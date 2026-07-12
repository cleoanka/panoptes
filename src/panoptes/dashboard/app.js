/* Panoptes operations console.
 *
 * Classic script on purpose: ES modules break under file:// and naive static
 * mounts. Everything is same-origin. Auth: fetch() carries X-API-Key; <img>
 * (MJPEG) and WebSocket cannot set headers, so they carry ?api_key= instead.
 * All dynamic text goes through textContent / createElement — innerHTML is
 * only ever fed constant icon markup.
 */
(() => {
  'use strict';

  // ----------------------------------------------------------- constants
  const KEY_STORAGE = 'panoptes.apiKey';
  const PREVIEW_STORAGE = 'panoptes.previewOff';
  const FEED_MAX = 300;          // rows kept in the DOM
  const SEEN_MAX = 4000;         // dedup set bound
  const STREAM_POLL_MS = 3000;
  const HEALTH_POLL_MS = 5000;
  const DRAWER_POLL_MS = 5000;
  const WS_DELAY_MIN = 1000;
  const WS_DELAY_MAX = 30000;
  const PREVIEW_RETRY_MS = 5000;
  // 1x1 transparent GIF: assigning it aborts an open MJPEG connection,
  // which removeAttribute('src') does not reliably do.
  const BLANK_IMG =
    'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

  // Must track panoptes.core.events.EventType. Unknown types arriving on the
  // wire self-register with a neutral color so nothing is silently dropped.
  const EVENT_TYPES = [
    ['track_started', 'track started', '#5f8fc4'],
    ['track_finished', 'track finished', '#4a6f96'],
    ['line_crossed', 'line crossed', '#3fae8e'],
    ['zone_entered', 'zone entered', '#3aa6a0'],
    ['zone_exited', 'zone exited', '#348e88'],
    ['zone_dwell', 'zone dwell', '#c9a22c'],
    ['speeding', 'speeding', '#d97a3d'],
    ['wrong_way', 'wrong way', '#d9534a'],
    ['stopped_vehicle', 'stopped vehicle', '#c9853f'],
    ['plate_read', 'plate read', '#9a86d9'],
    ['watchlist_hit', 'watchlist hit', '#d94a76'],
    ['rule_triggered', 'rule triggered', '#e8a33d'],
    ['stream_started', 'stream started', '#77828f'],
    ['stream_ended', 'stream ended', '#5c6773'],
    ['stream_error', 'stream error', '#d9534a'],
  ];
  const UNKNOWN_TYPE_COLOR = '#8b96a5';

  const ICON_CARET =
    '<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true">' +
    '<path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" ' +
    'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const ICON_TICK =
    '<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true">' +
    '<path d="M2.5 6.5 5 9l4.5-5.5" fill="none" stroke="currentColor" ' +
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const ICON_X =
    '<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true">' +
    '<path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" ' +
    'stroke-width="1.5" stroke-linecap="round"/></svg>';

  // ----------------------------------------------------------- dom utils
  const $ = (sel, root) => (root || document).querySelector(sel);

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function icon(markup, cls) {
    const s = el('span', cls || 'icon');
    s.innerHTML = markup; // constant strings only — never data-driven
    return s;
  }

  const pad2 = (n) => String(n).padStart(2, '0');

  function fmtClock(unixSeconds) {
    if (unixSeconds == null || !isFinite(unixSeconds)) return '—';
    const d = new Date(unixSeconds * 1000);
    return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }

  function fmtNum(v, digits) {
    const n = Number(v);
    return v == null || !isFinite(n) ? '—' : n.toFixed(digits == null ? 1 : digits);
  }

  const prettify = (t) => String(t).replace(/_/g, ' ');

  // localStorage throws in some embedded/file:// contexts — degrade to memory.
  const mem = {};
  function lsGet(key) {
    try { return localStorage.getItem(key); } catch (_e) { return mem[key] ?? null; }
  }
  function lsSet(key, value) {
    try { localStorage.setItem(key, value); } catch (_e) { mem[key] = value; }
  }

  // ----------------------------------------------------------- state
  const state = {
    key: lsGet(KEY_STORAGE) || '',
    streams: new Map(),     // id -> {id, data, dom, imgSrc, retry, lastErr, drawerOpen}
    previewOff: new Set(),
    typeMeta: new Map(),    // type -> {label, color, on, count, chip, countEl}
    seen: new Set(),
    feedTotal: 0,
    ws: null,
    wsTimer: null,
    wsDelay: WS_DELAY_MIN,
    apiState: null,
  };
  try {
    const raw = JSON.parse(lsGet(PREVIEW_STORAGE) || '[]');
    if (Array.isArray(raw)) raw.forEach((id) => state.previewOff.add(String(id)));
  } catch (_e) { /* corrupt persisted value — ignore */ }

  // ----------------------------------------------------------- auth + urls
  function authHeaders() {
    return state.key ? { 'X-API-Key': state.key } : {};
  }
  function keyQuery(sep) {
    return state.key ? (sep || '?') + 'api_key=' + encodeURIComponent(state.key) : '';
  }
  function mediaUrl(snapshotPath) {
    if (!snapshotPath) return null;
    let p = String(snapshotPath).replace(/^\/+/, '');
    if (p.startsWith('media/')) p = p.slice('media/'.length);
    return '/media/' + p.split('/').map(encodeURIComponent).join('/') + keyQuery();
  }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ headers: authHeaders(), cache: 'no-store' }, opts));
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.text()).slice(0, 160); } catch (_e) { /* body gone */ }
      const err = new Error('HTTP ' + res.status + (detail ? ' — ' + detail : ''));
      err.status = res.status;
      throw err;
    }
    if (res.status === 204) return null;
    return res.json();
  }

  // ----------------------------------------------------------- banners
  const bannerLast = new Map(); // message -> ts, to suppress repeat spam
  function banner(msg, kind) {
    const now = Date.now();
    if ((bannerLast.get(msg) || 0) > now - 10000) return;
    bannerLast.set(msg, now);
    const b = el('div', 'banner ' + (kind === 'warn' ? 'warn' : 'error'));
    b.append(el('span', 'banner-msg', msg));
    const x = el('button', 'banner-x');
    x.type = 'button';
    x.setAttribute('aria-label', 'Dismiss');
    x.append(icon(ICON_X));
    x.addEventListener('click', () => b.remove());
    b.append(x);
    $('#banners').append(b);
    setTimeout(() => b.remove(), 12000);
  }

  // ----------------------------------------------------------- header
  function setDot(id, cls, title) {
    const d = $(id);
    d.className = 'dot' + (cls ? ' ' + cls : '');
    if (title) d.title = title;
  }

  function setApiState(s) {
    if (s === state.apiState) return;
    state.apiState = s;
    if (s === 'ok') setDot('#dot-api', 'ok', 'API reachable');
    else if (s === 'auth') {
      setDot('#dot-api', 'warn', 'API rejected the key');
      banner('API rejected the request — check the API key in the header.', 'warn');
    } else {
      setDot('#dot-api', 'err', 'API unreachable');
      banner('API unreachable — is the Panoptes server running?');
    }
  }

  let healthBusy = false;
  async function pollHealth() {
    if (healthBusy) return; // slow/timing-out API must not stack requests
    healthBusy = true;
    try {
      await api('/api/v1/system/health');
      setApiState('ok');
    } catch (e) {
      setApiState(e.status === 401 || e.status === 403 ? 'auth' : 'down');
    } finally {
      healthBusy = false;
    }
  }

  function startClock() {
    const node = $('#clock');
    const tick = () => { node.textContent = fmtClock(Date.now() / 1000); };
    tick();
    setInterval(tick, 1000);
  }

  function wireKeyField() {
    const input = $('#api-key');
    const vis = $('#key-vis');
    input.value = state.key;
    input.addEventListener('change', () => {
      const v = input.value.trim();
      if (v === state.key) return;
      state.key = v;
      lsSet(KEY_STORAGE, v);
      // Everything that embeds the key must be rebuilt.
      state.streams.forEach((entry) => { entry.imgSrc = null; entry.lastErr = 0; });
      reconnectWs();
      pollHealth();
      refreshStreams();
      loadSystem();
    });
    vis.addEventListener('click', () => {
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      vis.classList.toggle('on', show);
      vis.setAttribute('aria-pressed', String(show));
    });
  }

  // ----------------------------------------------------------- streams
  function isRunning(stateName) {
    return ['running', 'started', 'active', 'live'].includes(stateName);
  }
  function badgeClass(stateName) {
    if (isRunning(stateName)) return 'st-running';
    if (['error', 'failed', 'crashed'].includes(stateName)) return 'st-error';
    if (['starting', 'connecting', 'reconnecting', 'opening'].includes(stateName)) return 'st-pending';
    return 'st-stopped';
  }

  function normalizeStreams(payload) {
    // Accepted shapes: [ {...} ], { streams: [...] }, { streams: {id: {...}} },
    // { id: {...} } — the API schema is owned by another module, so be liberal.
    let src = payload;
    if (src && typeof src === 'object' && !Array.isArray(src)
        && src.streams && typeof src.streams === 'object') {
      src = src.streams;
    }
    let list = [];
    if (Array.isArray(src)) list = src;
    else if (src && typeof src === 'object') {
      list = Object.entries(src).map(([k, v]) =>
        v && typeof v === 'object' ? Object.assign({ id: k }, v) : { id: k });
    }
    return list
      .filter((s) => s && typeof s === 'object')
      .map((s) => {
        const id = s.id ?? s.stream_id ?? s.name;
        const tracks = s.active_tracks ?? s.tracks ?? s.track_count ?? s.n_tracks;
        return {
          id: String(id),
          name: String(s.name ?? id),
          state: String(s.state ?? s.status ?? 'unknown').toLowerCase(),
          fps: s.fps,
          tracks,
        };
      });
  }

  function previewSrc(entry) {
    let url = '/api/v1/streams/' + encodeURIComponent(entry.id) + '/preview.mjpeg' + keyQuery();
    // Cache-buster forces the browser to reopen the multipart connection
    // after an error or a manual start/stop; absent otherwise so polling
    // never restarts a healthy MJPEG stream.
    if (entry.retry) url += (url.includes('?') ? '&' : '?') + 'r=' + entry.retry;
    return url;
  }

  function applyPreview(entry) {
    const { img, previewOffEl } = entry.dom;
    const running = isRunning(entry.data.state);
    const enabled = !state.previewOff.has(entry.id);
    const cooling = entry.lastErr && Date.now() - entry.lastErr < PREVIEW_RETRY_MS;
    if (running && enabled && !cooling) {
      const src = previewSrc(entry);
      if (entry.imgSrc !== src) {
        entry.imgSrc = src;
        img.src = src;
      }
      img.style.display = '';
      previewOffEl.style.display = 'none';
    } else {
      if (entry.imgSrc) {
        entry.imgSrc = null;
        img.src = BLANK_IMG;
      }
      img.style.display = 'none';
      previewOffEl.style.display = '';
      previewOffEl.textContent = !running ? 'no signal'
        : !enabled ? 'preview off'
        : 'preview unavailable — retrying';
    }
  }

  async function streamAction(entry, action, btn) {
    btn.disabled = true;
    try {
      await api('/api/v1/streams/' + encodeURIComponent(entry.id) + '/' + action, { method: 'POST' });
      entry.lastErr = 0;
      entry.retry++;
      await refreshStreams();
    } catch (e) {
      banner('Stream "' + entry.id + '" ' + action + ' failed: ' + e.message);
      btn.disabled = false;
    }
  }

  async function loadAnalytics(entry) {
    try {
      const data = await api('/api/v1/streams/' + encodeURIComponent(entry.id) + '/analytics');
      entry.dom.drawerBody.replaceChildren(renderNested(data, 0));
    } catch (e) {
      entry.dom.drawerBody.replaceChildren(el('div', 'muted', 'counters unavailable — ' + e.message));
    }
  }

  function buildStreamCard(id) {
    const root = el('article', 'stream-card');

    const head = el('div', 'stream-head');
    const title = el('div', 'stream-title');
    const name = el('h3', 'stream-name');
    const sid = el('div', 'stream-id', id);
    title.append(name, sid);
    const badge = el('span', 'badge');
    head.append(title, badge);

    const preview = el('div', 'preview');
    const img = el('img');
    img.alt = 'Live preview of stream ' + id;
    const previewOffEl = el('div', 'preview-off', 'no signal');
    preview.append(img, previewOffEl);

    const stats = el('div', 'stream-stats');
    const fpsWrap = el('span', null, 'fps');
    const fps = el('span', 'num', '—');
    fpsWrap.append(fps);
    const trWrap = el('span', null, 'tracks');
    const tracks = el('span', 'num', '—');
    trWrap.append(tracks);
    stats.append(fpsWrap, trWrap);

    const actions = el('div', 'stream-actions');
    const btnStart = el('button', 'btn btn-accent', 'start');
    btnStart.type = 'button';
    const btnStop = el('button', 'btn', 'stop');
    btnStop.type = 'button';
    const btnPreview = el('button', 'btn btn-toggle');
    btnPreview.type = 'button';
    btnPreview.append(icon(ICON_TICK, 'tick'), el('span', null, 'preview'));
    const btnDrawer = el('button', 'btn drawer-btn');
    btnDrawer.type = 'button';
    btnDrawer.append(el('span', null, 'counters'), icon(ICON_CARET, 'caret'));
    actions.append(btnStart, btnStop, btnPreview, btnDrawer);

    const drawer = el('div', 'drawer');
    const inner = el('div', 'drawer-inner');
    const drawerBody = el('div', 'drawer-pad');
    inner.append(drawerBody);
    drawer.append(inner);

    root.append(head, preview, stats, actions, drawer);

    const entry = {
      id,
      data: { state: 'unknown' },
      imgSrc: null,
      retry: 0,
      lastErr: 0,
      drawerOpen: false,
      dom: { root, name, badge, img, previewOffEl, fps, tracks, btnStart, btnStop, btnPreview, btnDrawer, drawer, drawerBody },
    };

    img.addEventListener('error', () => {
      if (!entry.imgSrc) return; // blanking the src also fires error in some engines
      entry.imgSrc = null;
      entry.lastErr = Date.now();
      entry.retry++;
      applyPreview(entry);
    });
    btnStart.addEventListener('click', () => streamAction(entry, 'start', btnStart));
    btnStop.addEventListener('click', () => streamAction(entry, 'stop', btnStop));
    btnPreview.addEventListener('click', () => {
      if (state.previewOff.has(id)) state.previewOff.delete(id);
      else state.previewOff.add(id);
      lsSet(PREVIEW_STORAGE, JSON.stringify([...state.previewOff]));
      btnPreview.classList.toggle('on', !state.previewOff.has(id));
      applyPreview(entry);
    });
    btnDrawer.addEventListener('click', () => {
      entry.drawerOpen = !entry.drawerOpen;
      drawer.classList.toggle('open', entry.drawerOpen);
      btnDrawer.classList.toggle('open', entry.drawerOpen);
      if (entry.drawerOpen) {
        drawerBody.replaceChildren(el('div', 'muted', 'loading counters…'));
        loadAnalytics(entry);
      }
    });

    btnPreview.classList.toggle('on', !state.previewOff.has(id));
    return entry;
  }

  function updateStreamCard(entry, s) {
    entry.data = s;
    const d = entry.dom;
    d.name.textContent = s.name;
    d.badge.textContent = s.state;
    d.badge.className = 'badge ' + badgeClass(s.state);
    d.fps.textContent = fmtNum(s.fps, 1);
    d.tracks.textContent = s.tracks == null ? '—' : String(s.tracks);
    // Errored streams keep both actions live: the worker may still hold the
    // source, so the operator must be able to force either transition.
    const bc = badgeClass(s.state);
    d.btnStart.disabled = bc === 'st-running' || bc === 'st-pending';
    d.btnStop.disabled = bc === 'st-stopped';
    applyPreview(entry);
  }

  let streamsBusy = false;
  async function refreshStreams() {
    if (streamsBusy) return;
    streamsBusy = true;
    let payload;
    try {
      payload = await api('/api/v1/streams');
    } catch (_e) {
      return; // health poll owns the unreachable banner
    } finally {
      streamsBusy = false;
    }
    const list = normalizeStreams(payload);
    const host = $('#stream-list');
    const seen = new Set();
    for (const s of list) {
      seen.add(s.id);
      let entry = state.streams.get(s.id);
      if (!entry) {
        entry = buildStreamCard(s.id);
        state.streams.set(s.id, entry);
        host.append(entry.dom.root); // stable order: server order on first sight
      }
      updateStreamCard(entry, s);
    }
    for (const [id, entry] of [...state.streams]) {
      if (!seen.has(id)) {
        entry.dom.root.remove();
        state.streams.delete(id);
      }
    }
    $('#streams-count').textContent = String(list.length);
    $('#streams-empty').hidden = list.length > 0;
  }

  function pollOpenDrawers() {
    state.streams.forEach((entry) => {
      if (entry.drawerOpen) loadAnalytics(entry);
    });
  }

  // ------------------------------------------------- nested dict renderer
  // Analytics summaries and system info have server-defined shapes; render
  // any JSON-ish nesting rather than assuming one.
  function renderNested(value, depth) {
    if (value == null || typeof value !== 'object') {
      return el('div', 'kv-row muted', String(value ?? 'null'));
    }
    if (Array.isArray(value) || depth > 4) {
      const blob = el('div', 'kv-blob');
      blob.textContent = JSON.stringify(value, null, 2);
      return blob;
    }
    const keys = Object.keys(value);
    const wrap = el('div');
    if (!keys.length) {
      wrap.append(el('div', 'muted', 'nothing recorded yet'));
      return wrap;
    }
    for (const k of keys) {
      const v = value[k];
      if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
        const section = el('div', 'kv-section');
        section.append(el('div', 'kv-head', prettify(k)));
        section.append(renderNested(v, depth + 1));
        wrap.append(section);
      } else {
        const row = el('div', 'kv-row');
        row.append(el('span', 'kv-key', prettify(k)));
        const val = Array.isArray(v) ? JSON.stringify(v) : String(v ?? '—');
        row.append(el('span', 'kv-val num', val));
        wrap.append(row);
      }
    }
    return wrap;
  }

  // ----------------------------------------------------------- event feed
  function applyFilters() {
    const feed = $('#feed');
    for (const row of feed.children) {
      const meta = state.typeMeta.get(row.dataset.type);
      row.hidden = !(meta && meta.on);
    }
  }

  function registerType(type, label, color) {
    let meta = state.typeMeta.get(type);
    if (meta) return meta;
    meta = { label, color, on: true, count: 0, chip: null, countEl: null };

    const chip = el('button', 'chip on');
    chip.type = 'button';
    chip.setAttribute('aria-pressed', 'true');
    const dot = el('span', 'chip-dot');
    dot.style.background = color;
    const countEl = el('span', 'chip-count num', '0');
    chip.append(icon(ICON_TICK, 'chip-tick'), dot, el('span', null, label), countEl);
    chip.addEventListener('click', () => {
      meta.on = !meta.on;
      chip.classList.toggle('on', meta.on);
      chip.setAttribute('aria-pressed', String(meta.on));
      applyFilters();
    });
    $('#chips').append(chip);

    meta.chip = chip;
    meta.countEl = countEl;
    state.typeMeta.set(type, meta);
    return meta;
  }

  function setAllFilters(on) {
    state.typeMeta.forEach((meta) => {
      meta.on = on;
      meta.chip.classList.toggle('on', on);
      meta.chip.setAttribute('aria-pressed', String(on));
    });
    applyFilters();
  }

  function summarize(ev) {
    const d = ev.data || {};
    const bits = [];
    switch (ev.type) {
      case 'plate_read':
        bits.push(d.plate || d.text || d.plate_text);
        if (d.confidence != null) bits.push('conf ' + fmtNum(d.confidence, 2));
        break;
      case 'watchlist_hit': {
        bits.push(d.plate || d.text || d.plate_text);
        const wl = d.watchlist || d.watchlist_id;
        if (wl) bits.push('list: ' + wl);
        break;
      }
      case 'speeding':
        if (d.speed_kmh != null) bits.push(fmtNum(d.speed_kmh, 0) + ' km/h');
        if (d.limit_kmh != null) bits.push('limit ' + fmtNum(d.limit_kmh, 0));
        break;
      case 'line_crossed':
        bits.push(d.line || d.line_id, d.direction);
        break;
      case 'zone_entered':
      case 'zone_exited':
        bits.push(d.zone || d.zone_id);
        break;
      case 'zone_dwell':
        bits.push(d.zone || d.zone_id);
        if (d.dwell_s != null) bits.push(fmtNum(d.dwell_s, 0) + 's dwell');
        break;
      case 'rule_triggered':
        bits.push(ev.rule_id || d.rule || d.rule_id);
        break;
      case 'track_finished':
        if (d.duration_s != null) bits.push(fmtNum(d.duration_s, 1) + 's');
        if (d.distance_m != null) bits.push(fmtNum(d.distance_m, 0) + ' m');
        if (d.avg_speed_kmh != null) bits.push(fmtNum(d.avg_speed_kmh, 0) + ' km/h avg');
        if (d.plate) bits.push(String(d.plate));
        break;
      case 'stream_error':
        bits.push(d.error || d.message || d.detail);
        break;
      default:
        break;
    }
    const cls = ev.vehicle_class || d.vehicle_class || d.class;
    if (cls) bits.unshift(String(cls));
    if (ev.track_id != null) bits.unshift('#' + ev.track_id);
    const text = bits.filter((b) => b != null && b !== '').join(' · ');
    return text || prettify(String(ev.type));
  }

  function buildDetail(ev, host) {
    const pre = el('pre', 'ev-json');
    pre.textContent = JSON.stringify(ev, null, 2);
    host.append(pre);
    const url = mediaUrl(ev.snapshot_path);
    if (url) {
      const img = el('img', 'ev-snap');
      img.alt = 'Event snapshot';
      img.loading = 'lazy';
      img.addEventListener('error', () => {
        img.replaceWith(el('div', 'muted', 'snapshot not retrievable: ' + ev.snapshot_path));
      });
      img.src = url;
      host.append(img);
    }
  }

  function buildEventRow(ev, meta) {
    const wrap = el('div', 'ev');
    wrap.dataset.type = String(ev.type);
    wrap.style.borderLeftColor = meta.color;

    const row = el('button', 'ev-row');
    row.type = 'button';
    const typeEl = el('span', 'ev-type', meta.label);
    typeEl.style.color = meta.color;
    row.append(
      el('span', 'ev-time num', fmtClock(ev.wall_ts)),
      typeEl,
      el('span', 'ev-stream', ev.stream_id || '—'),
      el('span', 'ev-summary', summarize(ev)),
      icon(ICON_CARET, 'caret'),
    );

    const drawer = el('div', 'drawer');
    const inner = el('div', 'drawer-inner');
    const detail = el('div', 'ev-detail');
    inner.append(detail);
    drawer.append(inner);

    let built = false;
    row.addEventListener('click', () => {
      const open = !wrap.classList.contains('open');
      wrap.classList.toggle('open', open);
      drawer.classList.toggle('open', open);
      if (open && !built) {
        built = true;
        buildDetail(ev, detail);
      }
    });

    wrap.append(row, drawer);
    return wrap;
  }

  function ingest(raw) {
    const ev = raw && typeof raw === 'object' ? (typeof raw.type === 'string' ? raw : raw.event) : null;
    if (!ev || typeof ev.type !== 'string') return;
    if (ev.id) {
      if (state.seen.has(ev.id)) return;
      if (state.seen.size >= SEEN_MAX) state.seen.clear();
      state.seen.add(ev.id);
    }
    const meta = registerType(ev.type, prettify(ev.type), UNKNOWN_TYPE_COLOR);
    meta.count++;
    meta.countEl.textContent = String(meta.count);

    const feed = $('#feed');
    const row = buildEventRow(ev, meta);
    row.hidden = !meta.on;
    feed.prepend(row);
    while (feed.children.length > FEED_MAX) feed.lastElementChild.remove();

    state.feedTotal++;
    $('#feed-total').textContent = String(state.feedTotal);
    $('#feed-empty').hidden = true;
  }

  async function backfillEvents() {
    try {
      const payload = await api('/api/v1/events?limit=100');
      const list = Array.isArray(payload)
        ? payload
        : (payload && (payload.events || payload.items || payload.results)) || [];
      // ingest() prepends, so feed ascending order to end newest-on-top.
      list
        .slice()
        .sort((a, b) => (a.wall_ts || 0) - (b.wall_ts || 0))
        .forEach(ingest);
    } catch (_e) {
      // history endpoint down — the live feed still works on its own
    }
  }

  // ----------------------------------------------------------- websocket
  function setFeedState(s) {
    if (s === 'ok') setDot('#dot-feed', 'ok', 'Event feed connected');
    else if (s === 'connecting') setDot('#dot-feed', 'connecting', 'Event feed connecting');
    else if (s === 'off') setDot('#dot-feed', '', 'No server origin (opened from file)');
    else setDot('#dot-feed', 'err', 'Event feed disconnected — reconnecting');
  }

  function wsUrl() {
    if (!location.host) return null; // file:// — nothing to connect to
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return proto + '//' + location.host + '/api/v1/events/ws' + keyQuery();
  }

  function connectWs() {
    clearTimeout(state.wsTimer);
    const url = wsUrl();
    if (!url) {
      setFeedState('off');
      return;
    }
    setFeedState('connecting');
    let sock;
    try {
      sock = new WebSocket(url);
    } catch (_e) {
      setFeedState('down');
      scheduleReconnect();
      return;
    }
    state.ws = sock;
    sock.onopen = () => {
      state.wsDelay = WS_DELAY_MIN;
      setFeedState('ok');
    };
    sock.onmessage = (m) => {
      try { ingest(JSON.parse(m.data)); } catch (_e) { /* non-JSON frame */ }
    };
    sock.onclose = () => {
      if (state.ws !== sock) return; // superseded by reconnectWs()
      state.ws = null;
      setFeedState('down');
      scheduleReconnect();
    };
    sock.onerror = () => {
      try { sock.close(); } catch (_e) { /* already closed */ }
    };
  }

  function scheduleReconnect() {
    clearTimeout(state.wsTimer);
    state.wsTimer = setTimeout(connectWs, state.wsDelay);
    state.wsDelay = Math.min(state.wsDelay * 1.8, WS_DELAY_MAX);
  }

  function reconnectWs() {
    const old = state.ws;
    state.ws = null; // disarm old onclose before closing
    if (old) {
      try { old.close(); } catch (_e) { /* already closed */ }
    }
    clearTimeout(state.wsTimer);
    state.wsDelay = WS_DELAY_MIN;
    connectWs();
  }

  // ----------------------------------------------------------- plates
  function plateRow(r) {
    const tr = el('tr');
    const plate = el('td', 'plate', String(r.plate ?? r.text ?? r.plate_text ?? '—'));
    const stream = el('td', null, String(r.stream_id ?? r.stream ?? '—'));
    const track = el('td', 'ta-r num', r.track_id == null ? '—' : String(r.track_id));
    const conf = el('td', 'ta-r num', fmtNum(r.confidence, 2));
    const validVal = r.valid;
    const valid = el(
      'td',
      validVal == null ? null : validVal ? 'valid-yes' : 'valid-no',
      validVal == null ? '—' : validVal ? 'yes' : 'no',
    );
    const country = el('td', null, String(r.country ?? '—'));
    const time = el('td', 'num', fmtClock(r.wall_ts ?? r.ts ?? r.timestamp));
    tr.append(plate, stream, track, conf, valid, country, time);
    return tr;
  }

  function wirePlates() {
    const form = $('#plate-form');
    const status = $('#plates-status');
    const table = $('#plates-table');
    const body = $('#plates-body');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const q = $('#plate-q').value.trim();
      status.hidden = false;
      status.textContent = 'searching…';
      table.hidden = true;
      try {
        const payload = await api('/api/v1/plates?q=' + encodeURIComponent(q) + '&limit=100');
        const list = Array.isArray(payload)
          ? payload
          : (payload && (payload.plates || payload.items || payload.results)) || [];
        $('#plates-count').textContent = String(list.length);
        body.replaceChildren(...list.map(plateRow));
        table.hidden = list.length === 0;
        status.hidden = list.length > 0;
        if (!list.length) status.textContent = 'no plate reads match "' + q + '"';
      } catch (err) {
        status.textContent = 'search failed — ' + err.message;
        banner('Plate search failed: ' + err.message);
      }
    });
  }

  // ----------------------------------------------------------- system
  async function loadSystem() {
    const host = $('#system-body');
    try {
      const info = await api('/api/v1/system/info');
      host.replaceChildren(renderNested(info, 0));
    } catch (e) {
      host.replaceChildren(el('div', 'empty', 'system info unavailable — ' + e.message));
    }
  }

  // ----------------------------------------------------------- init
  function init() {
    startClock();
    wireKeyField();
    wirePlates();
    for (const [type, label, color] of EVENT_TYPES) registerType(type, label, color);
    $('#chips-all').addEventListener('click', () => setAllFilters(true));
    $('#chips-none').addEventListener('click', () => setAllFilters(false));
    $('#system-refresh').addEventListener('click', loadSystem);

    setFeedState('connecting');
    pollHealth();
    refreshStreams();
    loadSystem();
    backfillEvents().then(connectWs);

    setInterval(pollHealth, HEALTH_POLL_MS);
    setInterval(refreshStreams, STREAM_POLL_MS);
    setInterval(pollOpenDrawers, DRAWER_POLL_MS);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
