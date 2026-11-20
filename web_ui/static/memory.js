// Experience-database panel on the run page (web_ui/static/interactive.html), loaded when the server runs with --memory-dir:
// Save to database for the current or a recent run, nearest start from the database (warm start), replay. The Database tab
// (/database) lists, inspects and manages saved runs. It plugs into the page through its extension hooks (window.slicUI:
// on('event'|'request'|'stop'), ready, selectTask) and uses the page's globals (handleEvent, resetUI, setStatus, goalSpecs,
// stepsData, readTolerances, curNode, msg) for replay.
// URL parameters: ?replay=<run_id> replays a run; ?warm=<run_id>:<step> selects that run's task and fills its design as the start.
(function () {
  if (!window.slicUI) { console.error('memory panel: the page has no extension hooks (window.slicUI)'); return; }
  const qs = s => document.querySelector(s);
  const num = id => { const x = parseFloat(qs(id).value); return isNaN(x) ? null : x; };
  // everything the panel reads from or does to the run page
  const H = {
    panelParent: () => qs('.sidebar'),
    navParent: () => qs('.header nav'),
    // resolves once the page has loaded its task (and reselected the last one)
    ready: () => Promise.race([window.slicUI.ready, new Promise((_, rej) =>
      setTimeout(() => rej(new Error('the page did not finish loading its tasks')), 60000))]),
    // select a stored run's task (node, topology, task index) the way the page's own controls do
    selecting: false,
    selectTask: async ref => {
      H.selecting = true;
      try { await window.slicUI.selectTask(ref); } finally { H.selecting = false; }
    },
    onRunRequest: cb => window.slicUI.on('request', cb),
    taskRequest: () => ({ topology: qs('#selTopology').value, node: curNode() }),
    // the goal as startInference reads it
    readGoal: () => {
      const g = {};
      [['gain_min', '#goalGain'], ['power_max', '#goalPower'], ['bandwidth_min', '#goalBW'],
       ['phase_margin_min_deg', '#goalPM'], ['phase_margin_max_deg', '#goalPMMax']].forEach(([k, id]) => {
        const v = num(id); if (v != null) g[k] = v;
      });
      return g;
    },
    readTolerances: () => readTolerances(),   // fractions, or null = the protocol's (throws on invalid input)
    // operating conditions for the metadata filter: the supply in the starting-sizes editor (the agent does not change it)
    readConditions: () => {
      const i = document.querySelector('#paramEditor input[data-param="vdd"]');
      return i && i.value.trim() !== '' ? { vdd: i.value.trim() } : null;
    },
    // put a stored design into the starting-sizes editor; every key must exist there (same topology)
    fillStart: state => {
      const inputs = {};
      document.querySelectorAll('#paramEditor input[data-param]').forEach(i => { inputs[i.dataset.param] = i; });
      const missing = Object.keys(state).filter(k => !(k in inputs));
      if (missing.length) throw new Error('the editor has no field for ' + missing.slice(0, 5).join(', ') + ' (select the same topology)');
      Object.values(inputs).forEach(i => { i.value = ''; });   // empty = not sent; every stored key is filled below
      Object.entries(state).forEach(([k, v]) => { inputs[k].value = String(v); });
    },
    onTaskChange: cb => {
      ['#selNode', '#selCircuit', '#selTopology', '#taskIndex'].forEach(id => qs(id) && qs(id).addEventListener('change', cb));
      document.querySelectorAll('#inputSwitch button').forEach(b => b.addEventListener('click', cb));
    },
    onStartEdited: cb => { qs('#paramEditor').addEventListener('input', cb); },
    beginReplay: () => {
      resetUI(); setStatus('running');
      stepsData = []; bestNetlist = ''; bestStepIdx = -1;
      qs('#btnRun').disabled = true;
    },
    handle: evt => {
      if (evt.event === 'init') goalSpecs = Object.assign({}, evt.goal_specs || {});   // startInference sets it for live runs
      handleEvent(evt);
    },
    endReplay: (sawDone, note) => {
      qs('#btnRun').disabled = false;
      if (!sawDone) { setStatus('idle'); msg(note, false); }
    },
    message: (text, isErr) => msg(text, isErr),
  };
  const API = window.location.origin;
  const esc = s => { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; };
  const fmt = (v, d) => v == null || isNaN(v) ? 'n/a' : Number(v).toFixed(d);
  const ttlText = h => h == null ? 'kept until removed' : 'kept ' + h + ' h';
  let cfg = null, warm = null, replayCtrl = null, live = null, lastRun = null;

  const css = document.createElement('style');
  css.textContent = `
  #dmPanel{border-top:1px solid var(--border,#30363d);padding-top:12px;font-size:12px}
  #dmPanel .dm-row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0}
  #dmPanel button{font-size:11px;padding:3px 8px;border-radius:6px;border:1px solid var(--border,#30363d);background:var(--bg3,#21262d);color:var(--text,#c9d1d9);cursor:pointer}
  #dmPanel button:hover{border-color:var(--accent,#58a6ff)}
  #dmPanel button.dm-primary{background:var(--accent2,#1f6feb);border-color:var(--accent2,#1f6feb);color:#fff}
  #dmPanel select,#dmPanel input[type=text]{font-size:11px;background:var(--bg3,#21262d);color:var(--text,#c9d1d9);border:1px solid var(--border,#30363d);border-radius:6px;padding:2px 4px}
  #dmPanel .dm-item{border:1px solid var(--border,#30363d);border-radius:6px;padding:6px;margin:4px 0;background:var(--bg,#0d1117)}
  #dmPanel .dm-muted{color:var(--text2,#8b949e)}
  #dmPanel .dm-ok{color:var(--green,#3fb950)} #dmPanel .dm-bad{color:var(--red,#f85149)} #dmPanel .dm-est{color:var(--yellow,#d29922)}
  #dmWarm,#dmSave{border:1px dashed var(--accent,#58a6ff);border-radius:6px;padding:6px;margin:4px 0}
  #dmList{max-height:340px;overflow-y:auto}`;
  document.head.appendChild(css);

  const nav = H.navParent && H.navParent();
  if (nav) { const a = document.createElement('a'); a.href = '/database'; a.textContent = 'Database'; a.id = 'dmNavDb'; nav.appendChild(a); }

  const panel = document.createElement('div');
  panel.id = 'dmPanel';
  panel.innerHTML = `
    <div class="section-title">Experience database <a href="/database" style="text-transform:none;font-weight:400">open &rarr;</a></div>
    <div id="dmStats" class="dm-muted">loading ...</div>
    <div id="dmLive" class="dm-muted"></div>
    <div id="dmSave" class="hidden"></div>
    <div id="dmWarm" class="hidden"></div>
    <div class="dm-row">
      <button id="dmNearest" title="rank the database's designs of this topology/node against the goal and tolerances above">Find nearest start</button>
      <button id="dmUnsaved" title="runs recorded on this server but not saved; they are deleted after the retention time">Unsaved runs</button>
    </div>
    <details id="dmAdv"><summary class="dm-muted" style="cursor:pointer">ranking weights</summary>
      <div class="dm-row" id="dmWeights"></div>
      <div class="dm-muted">D = &Sigma; w<sub>j</sub> &middot; one-sided violation of spec j in decades (0.1 = a factor 1.26 outside its tolerated bound)</div>
    </details>
    <div class="dm-row"><span class="dm-muted">replay speed</span>
      <select id="dmSpeed"><option value="1">1x</option><option value="4" selected>4x</option><option value="20">20x</option><option value="0">instant</option></select>
      <button id="dmStopReplay" class="hidden">Stop replay</button>
    </div>
    <div id="dmNote" class="dm-muted"></div>
    <div id="dmList"></div>`;
  H.panelParent().appendChild(panel);
  const $ = s => panel.querySelector(s);
  const note = (t, err) => { $('#dmNote').innerHTML = t ? '<span class="' + (err ? 'dm-bad' : 'dm-muted') + '">' + esc(t) + '</span>' : ''; };

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    let j = null; try { j = await r.json(); } catch (_) { }
    if (!r.ok) throw new Error((j && j.detail) || (r.status + ' ' + r.statusText));
    return j;
  }
  const post = (url, body) => getJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  async function refreshStats() {
    try {
      cfg = await getJSON(API + '/api/memory/config');
      if (!$('#dmWeights').children.length) {
        $('#dmWeights').innerHTML = cfg.ui.tolerance_keys.map(k => '<label class="dm-muted">' + esc(k) +
          ' <input type="text" data-w="' + esc(k) + '" value="1" style="width:38px"></label>').join('');
      }
      const s = cfg.stats;
      $('#dmStats').textContent = s.saved_runs + ' runs in the database (' + s.saved_successful_runs + ' met their goal), ' +
        s.indexed_states + ' designs indexed; ' + s.unsaved_runs + ' unsaved (' + ttlText(s.unsaved_ttl_hours) + ')';
    } catch (e) { $('#dmStats').innerHTML = '<span class="dm-bad">database unavailable: ' + esc(e.message) + '</span>'; }
  }
  function metricsLine(m) {
    return (cfg ? cfg.ui.metrics : []).map(x => x.label + ' ' + fmt((m || {})[x.key], x.key === 'bandwidth_MHz' ? 3 : 1) + ' ' + x.unit).join(' | ');
  }

  // ---- save to database ------------------------------------------------------------------------------------------
  function offerSave(runId, result) {
    lastRun = runId;
    const el = $('#dmSave');
    el.classList.remove('hidden');
    el.innerHTML = '<div>Run <b>' + esc(runId) + '</b> ' + esc(result) + ' is <b>not saved</b>.</div>' +
      '<div class="dm-row"><input type="text" id="dmSaveLabel" placeholder="label (optional)" style="flex:1">' +
      '<button class="dm-primary" id="dmSaveBtn">Save to database</button><button id="dmSaveSkip">Not now</button></div>' +
      '<div class="dm-muted">Unsaved runs stay under "Unsaved runs" ' + (!cfg ? '' : cfg.stats.unsaved_ttl_hours == null ? 'until removed' :
        'for ' + cfg.stats.unsaved_ttl_hours + ' h, then are deleted') + ' (set on the Database page).</div>';
    $('#dmSaveBtn').onclick = () => saveRun(runId, $('#dmSaveLabel').value);
    $('#dmSaveSkip').onclick = () => { el.classList.add('hidden'); };
  }
  async function saveRun(runId, label) {
    try {
      await post(API + '/api/memory/runs/' + encodeURIComponent(runId) + '/save', label ? { label } : {});
      if (lastRun === runId) {
        $('#dmSave').innerHTML = '<span class="dm-ok">Saved ' + esc(runId) + ' to the database.</span> <a href="/database">open</a>';
      }
      note('Saved ' + runId + '.');
      refreshStats();
      if ($('#dmList').dataset.view === 'unsaved') unsavedList();
    } catch (e) { note('save failed: ' + e.message, true); }
  }
  async function unsavedList() {
    note('loading ...');
    try {
      const d = await getJSON(API + '/api/memory/runs?saved=false&limit=50');
      $('#dmList').dataset.view = 'unsaved';
      note(d.runs.length ? d.runs.length + ' unsaved run(s), newest first' : 'No unsaved runs.');
      $('#dmList').innerHTML = d.runs.map(r => {
        const res = r.status !== 'done' ? '<span class="dm-est">' + esc(r.status) + '</span>' :
          (r.success ? '<span class="dm-ok">goal met</span>' : '<span class="dm-bad">goal not met</span>');
        return '<div class="dm-item"><div>' + res + ' <span class="dm-muted">' + esc(r.created_utc) + ' UTC, ' + r.n_steps + ' turns</span></div>' +
          '<div class="dm-muted">' + esc(r.task_key) + (r.best_metrics ? ' | best: ' + esc(metricsLine(r.best_metrics)) : '') + '</div>' +
          '<div class="dm-row">' + (r.status === 'running' ? '<span class="dm-muted">running</span>' :
            '<button class="dm-primary" data-save="' + esc(r.run_id) + '">Save to database</button><button data-replay="' + esc(r.run_id) + '">Replay</button>') +
          '</div></div>';
      }).join('');
      $('#dmList').querySelectorAll('[data-save]').forEach(b => b.onclick = () => saveRun(b.dataset.save, ''));
      $('#dmList').querySelectorAll('[data-replay]').forEach(b => b.onclick = () => replay(b.dataset.replay));
    } catch (e) { note('unsaved runs: ' + e.message, true); }
  }

  // ---- warm start ------------------------------------------------------------------------------------------------
  function setWarm(w) {
    warm = w;
    const el = $('#dmWarm');
    if (!w) { el.classList.add('hidden'); el.innerHTML = ''; return; }
    el.classList.remove('hidden');
    el.innerHTML = 'Next run starts from <b>' + esc(w.run_id) + '</b> turn ' + (w.step < 0 ? 'start' : w.step + 1) +
      (w.edited ? ' <span class="dm-est">(sizes edited: not checked against the stored design)</span>' : '') +
      ' <button id="dmClearWarm">clear</button>';
    $('#dmClearWarm').onclick = () => setWarm(null);
  }
  H.onTaskChange(() => { if (warm && !H.selecting) { setWarm(null); note('Warm start cleared: the task changed.'); } });
  H.onStartEdited(() => { if (warm && !warm.edited) setWarm(Object.assign({}, warm, { edited: true })); });
  // the page calls this with every run request before it is sent: a chosen warm start is added to it
  H.onRunRequest(body => { if (warm) body.warm_start = { run_id: warm.run_id, step: warm.step }; });
  function useStart(runId, step, state) {
    H.fillStart(state);
    setWarm({ run_id: runId, step: step, edited: false });
    note('Starting sizes filled from the stored design; press Run.');
  }

  // ---- nearest start ---------------------------------------------------------------------------------------------
  async function nearest() {
    let tol;
    try { tol = H.readTolerances(); } catch (e) { note(e.message, true); return; }
    const goal = H.readGoal();
    if (!Object.keys(goal).length) { note('Set at least one goal field first.', true); return; }
    const weights = {};
    for (const i of panel.querySelectorAll('#dmWeights input[data-w]')) {
      const v = parseFloat(i.value);
      if (isNaN(v) || v < 0) { note('weight ' + i.dataset.w + ' must be a number >= 0', true); return; }
      weights[i.dataset.w] = v;
    }
    const conditions = H.readConditions ? H.readConditions() : null;
    note('ranking ...');
    try {
      const d = await post(API + '/api/memory/nearest', { task: H.taskRequest(), goal, tolerances: tol, conditions, weights, k: 5 });
      const f = d.funnel, ex = Object.entries(f.excluded_by_condition || {});
      const funnel = d.task_key + ' (saved, valid): ' + f.pool + ' designs' +
        (Object.keys(d.conditions || {}).length ? ' | conditions ' + Object.entries(d.conditions).map(([k, v]) => k + '=' + v).join(', ') +
          ': ' + (ex.length ? ex.map(([k, n]) => n + ' ' + k).join(', ') : 'none excluded') : '') +
        ' | ranked ' + f.ranked + ' (' + f.unique + ' distinct)';
      $('#dmList').dataset.view = 'nearest';
      if (!d.candidates.length) { note(f.pool ? funnel + ': nothing left after the condition filter.' : 'No saved designs for ' + d.task_key + ' yet.'); $('#dmList').innerHTML = ''; return; }
      note(funnel + '. ' + d.note);
      $('#dmList').innerHTML = d.candidates.map((c, i) => {
        const v = c.verdict, cls = v === 'exact_success' ? 'dm-ok' : v === 'estimated_pass' ? 'dm-est' : 'dm-bad';
        const fails = Object.entries(c.specs).filter(([, e]) => !e.pass)
          .map(([k, e]) => e.measurable ? k + ' ' + fmt(e.violation, 3) : k + ' unmeasurable');
        return '<div class="dm-item"><div>#' + (i + 1) + ' <b>D = ' + fmt(c.distance, 3) + '</b> <span class="' + cls + '">' + esc(v.replace('_', ' ')) + '</span>' +
          (fails.length ? ' <span class="dm-muted">misses ' + esc(fails.join(', ')) + '</span>'
                        : ' <span class="dm-muted">robustness ' + fmt(c.robustness, 3) + ' dec</span>') + '</div>' +
          '<div class="dm-muted">' + esc(metricsLine(c.metrics)) + '</div>' +
          '<div class="dm-muted">' + esc(c.run_id) + ' turn ' + (c.step < 0 ? 'start' : c.step + 1) + (c.run_label ? ' | ' + esc(c.run_label) : '') + '</div>' +
          '<div class="dm-row"><button data-use="' + i + '">Use as start</button><button data-replay="' + esc(c.run_id) + '">Replay run</button></div></div>';
      }).join('');
      $('#dmList').querySelectorAll('[data-use]').forEach(b => b.onclick = () => {
        const c = d.candidates[+b.dataset.use];
        try { useStart(c.run_id, c.step, c.state); } catch (e) { note(e.message, true); }
      });
      $('#dmList').querySelectorAll('[data-replay]').forEach(b => b.onclick = () => replay(b.dataset.replay));
    } catch (e) { note('nearest failed: ' + e.message, true); }
  }

  // ---- replay ----------------------------------------------------------------------------------------------------
  async function replay(id) {
    if (replayCtrl) replayCtrl.abort();
    const ctrl = new AbortController(); replayCtrl = ctrl;
    const speed = $('#dmSpeed').value;
    $('#dmStopReplay').classList.remove('hidden');
    H.beginReplay();
    H.message('Replaying ' + id + ' ...');
    let sawDone = false, why = 'Replay of ' + id + ' ended (the recorded run has no done event).';
    window.DM.replaying = true;
    try {
      const r = await fetch(API + '/api/memory/runs/' + encodeURIComponent(id) + '/replay?speed=' + speed, { signal: ctrl.signal });
      if (!r.ok) { let j = null; try { j = await r.json(); } catch (_) { } throw new Error((j && j.detail) || r.status); }
      const rd = r.body.getReader(), dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\n\n'); buf = parts.pop();
        for (const p of parts) {
          if (!p.startsWith('data: ')) continue;
          const e = JSON.parse(p.slice(6));
          if (e.event === 'memory') continue;
          if (e.event === 'done') sawDone = true;
          H.handle(e);
        }
      }
    } catch (e) {
      why = e.name === 'AbortError' ? 'Replay stopped.' : 'Replay failed: ' + e.message;
      sawDone = false;
    } finally {
      window.DM.replaying = false;
      if (replayCtrl === ctrl) replayCtrl = null;
      $('#dmStopReplay').classList.add('hidden');
      H.endReplay(sawDone, why);
    }
  }

  window.DM = {
    replaying: false,
    onMemoryEvent: e => {
      live = e.run_id;
      $('#dmSave').classList.add('hidden');
      $('#dmLive').innerHTML = 'Recording run <b>' + esc(e.run_id) + '</b> (not saved yet)' + (e.warm_start ? ', warm start from ' + esc(e.warm_start.run_id) : '');
    },
    onRunEnd: e => {
      if (window.DM.replaying || !live) return;
      const id = live; live = null;
      const res = e.event === 'done' ? (e.success ? '(goal met)' : '(goal not met)') : '(ended with an error)';
      $('#dmLive').innerHTML = 'Run <b>' + esc(id) + '</b> finished ' + esc(res);
      setTimeout(() => { refreshStats().then(() => offerSave(id, res)); }, 300);   // the server closes the run right after the last event
    },
    onRunStopped: () => {   // the user pressed Stop: the server keeps it as an aborted, unsaved run
      if (window.DM.replaying || !live) return;
      const id = live; live = null;
      $('#dmLive').innerHTML = 'Run <b>' + esc(id) + '</b> stopped';
      setTimeout(() => { refreshStats().then(() => offerSave(id, '(stopped)')); }, 1500);
    },
    replay,
  };
  $('#dmNearest').onclick = nearest;
  $('#dmUnsaved').onclick = unsavedList;
  $('#dmStopReplay').onclick = () => { if (replayCtrl) replayCtrl.abort(); };

  (async () => {
    await refreshStats();
    const q = new URLSearchParams(window.location.search);
    try {
      if (q.get('replay')) { await H.ready(); replay(q.get('replay')); }
      else if (q.get('warm')) {
        const m = q.get('warm').match(/^(.+):(-?\d+)$/);
        if (!m) throw new Error('warm= needs <run_id>:<step>');
        const d = await getJSON(API + '/api/memory/runs/' + encodeURIComponent(m[1]));
        const st = d.states.find(s => s.step === +m[2]);
        if (!st) throw new Error('run ' + m[1] + ' has no turn ' + m[2]);
        await H.ready();
        await H.selectTask(d.run.task_ref);
        useStart(m[1], +m[2], st.state);
      }
    } catch (e) { note(e.message, true); }
  })();
  // the memory event (run id) goes to the panel and not to the page; run ends and Stop are reported to the panel
  window.slicUI.on('event', evt => {
    if (evt && evt.event === 'memory') { window.DM.onMemoryEvent(evt); return false; }
    if (evt && (evt.event === 'done' || evt.event === 'error')) window.DM.onRunEnd(evt);
  });
  window.slicUI.on('stop', () => window.DM.onRunStopped());
})();
