'use strict';
// Presentation is a view of index.html. Content and figures stay in that file.
(() => {
  const $ = id => document.getElementById(id);
  const article = $('content');
  const app = window.Explainer;
  if (!article || !app) return;

  const source = selector => article.querySelector(selector) || document.querySelector(selector);
  const original = {
    title: source('h1'), tagline: source('.deck'), intro: source('.intro'),
    author: source('.author-byline'), brand: document.querySelector('.wordmark'),
    copyright: source('.copyright'), outline: source('.reading-map'),
    mlp: source('#precision .equation'),
    orientation: source('#orientation-panel'), resources: source('.source-grid'),
    support: source('.support-table'), code: source('.code-block')
  };
  const figures = Object.fromEntries([...article.querySelectorAll('figure[id]')].map(f => [f.id, f]));
  const paragraphs = Object.fromEntries([...article.querySelectorAll('section[id]')].map(section => [section.id, [...section.children].filter(n => ['P', 'DETAILS'].includes(n.tagName))]));
  paragraphs.motivation = [...source('#motivation').children].filter(node => node.tagName === 'P');

  // Selectors and camera windows define pacing. They contain no duplicate math,
  // prose, numerical values, figure markup, or example programs.
  const scenes = [
    { id: 'title', kind: 'cover', chapter: 'content' },
    { id: 'moe', figure: 'moe-figure', chapter: 'content', parts: ['moe-svg'] },
    { id: 'capabilities', figure: 'capability-figure', chapter: 'motivation', parts: ['capability-table', 'capability-notes'], setup: '#motivation-setup' },
    { id: 'system', figure: 'overview-figure', controller: 'overview', chapter: 'content', range: [0, 8], setup: '.diagram-key' },
    { id: 'assignments', figure: 'token-figure', controller: 'tokens', chapter: 'tokens', range: [0, 0] },
    { id: 'expert-rows', figure: 'token-figure', controller: 'tokens', chapter: 'tokens', range: [1, 1] },
    { id: 'return-slots', figure: 'token-figure', controller: 'tokens', chapter: 'tokens', range: [2, 3] },
    { id: 'quantization', figure: 'precision-figure', chapter: 'precision', parts: ['precision-svg', 'quant-detail'] },
    { id: 'quantization-axis', kind: 'orientation', chapter: 'precision' },
    { id: 'precision', figure: 'precision-figure', chapter: 'precision', parts: ['intermediate-table'], title: 'precision-reference', table: true },
    { id: 'allocation', figure: 'memory-figure', chapter: 'memory', parts: ['memory-policy-summary','consumption-svg', 'memory-stats'], setup: '#memory-shape', title: 'allocation', builds: ['#memory-examples [data-memory-example="saved"]','#memory-examples [data-memory-example="scratch"]','#memory-examples [data-memory-example="fixed"]'] },
    { id: 'memory-costs', figure: 'memory-figure', chapter: 'memory', parts: ['memory-svg'], setup: '#memory-config', title: 'costs' },
    { id: 'lifetimes', figure: 'recompute-figure', chapter: 'recompute', parts: ['recompute-svg', 'lifetime-legend'], prepare: 'lifetime', title: 'key-idea', setup: '.config-summary', bounds: [0,0,1440,690] },
    { id: 'layer-packing', figure: 'packing-figure', controller: 'packing', chapter: 'recompute', range: [0, 8], parts: ['packing-svg'], display: 'packing', title: 'key-idea', setup: '#packing-brief' },
    { id: 'gpu-tile', figure: 'kernel-figure', controller: 'kernel', chapter: 'kernels', range: [0, 5], title: 'key-idea', setup: '#tile-address' },
    { id: 'mega-pipeline', figure: 'fusion-figure', controller: 'fusion', chapter: 'kernels', range: [0, null], title: 'key-idea', setup: '#fusion-config' },
    { id: 'pipeline-prefix', figure: 'slots-figure', controller: 'slots', chapter: 'slots', range: [0, 31], actions: ['0F0','2B0','0F4','0B7'], slotDisplay: 'liveness', parts: ['liveness-svg'], setup: '#slots-config' },
    { id: 'slot-reuse', figure: 'slots-figure', controller: 'slots', chapter: 'slots', range: [2, 6], actions: ['2F0','2B0','0F3'], slotDisplay: 'reuse', camera: [0, 0, 1146, 406], pane: true, parts: ['slots-svg','slot-selection'] },
    { id: 'results-metrics', figure: 'results-metrics-figure', chapter: 'results', parts: ['metrics-overview'], setup: '#metrics-setup', title: 'key-idea' },
    { id: 'results-inference', figure: 'results-efficiency-figure', controller: 'efficiency', chapter: 'results', range: [0,1], parts: ['efficiency-svg','efficiency-table'], preset: '#result-presets [data-preset="inference"]', setup: '#efficiency-config', title: 'key-idea' },
    { id: 'results-training', figure: 'results-efficiency-figure', controller: 'efficiency', chapter: 'results', range: [0,1], parts: ['efficiency-svg','efficiency-table'], preset: '#result-presets [data-preset="training"]', setup: '#efficiency-config', title: 'key-idea' },
    { id: 'results-skew', figure: 'results-skew-figure', controller: 'skew', chapter: 'results', range: [0,2], parts: ['skew-svg','skew-table'], setup: '#skew-config', title: 'key-idea' },
    { id: 'results-width', figure: 'results-width-figure', controller: 'width', chapter: 'results', range: [0,2], parts: ['width-svg','width-table'], setup: '#width-config', title: 'key-idea' },
    { id: 'results-memory', figure: 'results-memory-figure', controller: 'result-memory', chapter: 'results', range: [0,2], parts: ['result-memory-svg','result-memory-table'], setup: '#result-memory-config', title: 'key-idea' },
    { id: 'configuration', kind: 'code', chapter: 'start', excerpt: 'config' },
    { id: 'expert-call', kind: 'code', chapter: 'start', excerpt: 'call' },
    { id: 'reading', kind: 'resources', chapter: 'start' }
  ];

  const sceneAliases = {'expert-math':'moe','dynamic-recompute':'layer-packing',offsets:'layer-packing','gather-tiles':'gpu-tile','tensor-cores':'gpu-tile','warp-groups':'mega-pipeline','chunk-readiness':'mega-pipeline'};
  /** Preserve links to examples that now share one complete slide. */
  function sceneIndex(id) { return scenes.findIndex(scene=>scene.id===(sceneAliases[id]||id)); }

  let shell, canvas, active = false, index = 0, build = 0, mounted = [], observer;
  let controlsOpen = false, notesOpen = false, fullView = false, zoomed = false;
  let updateQueued = false, savedFocus, slideFade, slideFadeTimer, speakerWindow, calibrating = false, exporting = false;
  const printMode = new URLSearchParams(location.search).get('view') === 'print';
  const savedCamera = new WeakMap();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  function cleanClone(node) {
    const clone = node.cloneNode(true);
    clone.removeAttribute('id');
    clone.querySelectorAll('[id]').forEach(n => n.removeAttribute('id'));
    clone.querySelectorAll('button').forEach(button=>button.replaceWith(document.createTextNode(button.textContent)));
    clone.querySelectorAll('a').forEach(a => { if (/^https?:/.test(a.href)) a.target = '_blank'; });
    return clone;
  }
  function move(node, target) {
    if (!node) return;
    const marker = document.createComment('presentation view');
    node.before(marker);
    target.append(node);
    mounted.push({ node, marker });
  }
  function restore() {
    observer?.disconnect();
    Object.values(app.controllers).forEach(c => c.stop());
    mounted.forEach(({ node }) => {
      node.querySelectorAll?.('svg[id]').forEach(svg => {
        app.finishTransition(svg);
        const old = savedCamera.get(svg);
        if (svg.id==='recompute-svg') svg.setAttribute('viewBox',app.state.retentionDisplay==='lifetime'?'0 0 1440 690':'0 0 1040 400');
        else if (old) svg.setAttribute('viewBox', old);
        svg.querySelectorAll('.deck-camera-clipped').forEach(n => n.classList.remove('deck-camera-clipped'));
        svg.classList.remove('deck-map');
        svg.style.removeProperty('--deck-svg-width');svg.style.removeProperty('--deck-svg-height');
        svg.querySelectorAll('[data-sticky-slot]').forEach(n => n.setAttribute('transform', 'translate(0,0)'));
      });
      node.querySelectorAll?.('.deck-reserved').forEach(clearReservation);
      node.querySelectorAll?.('.deck-part').forEach(n => n.classList.remove('deck-part'));
      node.classList.remove('deck-part');
    });
    for (const { node, marker } of mounted.reverse()) marker.replaceWith(node);
    mounted = [];
  }
  function sentence(text, count = 2) {
    try { return [...new Intl.Segmenter('en', { granularity: 'sentence' }).segment(text)].slice(0,count).map(s=>s.segment).join('').trim() || text; }
    catch { return text.match(/^.*?[.!?](?:\s|$)/)?.[0].trim() || text; }
  }
  function captionNode(scene) {
    if (!scene.figure) return null;
    return figures[scene.figure].querySelector('.phase-caption');
  }
  function titleFor(scene, current = false) {
    if (scene.kind === 'cover') return original.title.textContent;
    if (scene.kind === 'orientation') return source('#precision > h3').textContent;
    if (scene.kind === 'resources') return 'Execution modes and documentation';
    if (scene.kind === 'code') return scene.excerpt === 'config' ? 'dist_moe.Config' : (current&&app.state.code==='memory'?'dist_moe.plan_memory':'dist_moe.routed_experts');
    const figure = figures[scene.figure];
    if(current && scene.controller==='efficiency')return source(`[data-slide-summary="results-${app.state.resultMode}"]`).previousElementSibling.textContent;
    if (scene.title === 'precision-reference') return figure.querySelector('.inside-heading').textContent;
    if (scene.title === 'allocation') return figure.querySelector('.inside-heading').textContent;
    if (scene.title === 'costs') return figure.querySelectorAll('.inside-heading')[1].textContent;
    if (scene.controller === 'slots') return source(`[data-slide-summary="${scene.id}"]`).previousElementSibling.textContent;
    if (scene.title === 'chunks') return 'Chunk state';
    if (scene.title === 'key-idea') return source(`[data-slide-summary="${scene.id}"]`).previousElementSibling.textContent;
    if (scene.prepare) return document.querySelector(`#retention-display [data-view="${scene.prepare}"]`).textContent;
    if(!current && scene.controller && scene.id!=='system'){const caption=app.getStepCaption(scene.controller,scene.range[0]);if(caption)return caption[0];if(scene.controller==='slots')return figure.querySelector('h3').textContent+' · actions '+(scene.range[0]+1)+'–'+(scene.range[1]+1);}
    if (current && scene.controller && (scene.id !== 'system' || app.controllers.overview.phase > 0)) {
      const heading = captionNode(scene)?.querySelector('strong');
      if (heading) return heading.textContent.replace(/\.$/, '');
    }
    return figure.querySelector('h3').textContent;
  }
  function createShell() {
    shell = document.createElement('div');
    shell.id = 'presentation';
    shell.innerHTML = `
      <header class="deck-toolbar" aria-label="Presentation controls">
        <button id="deck-article" title="Return to the article">Article</button>
        <label class="deck-picker"><span class="sr-only">Slide</span><select id="deck-slide-select" aria-label="Choose a slide"></select></label>
        <div class="deck-tools">
          <button id="deck-export" title="Export every presentation step as a PDF page">Export PDF</button>
          <button id="deck-controls" aria-expanded="false" aria-controls="deck-controls-panel">Controls</button>
          <button id="deck-notes" aria-expanded="false" aria-controls="deck-notes-panel">Notes</button>
          <button id="deck-presenter" title="Open speaker notes in a separate window">Presenter</button>
          <button id="deck-fit" title="Toggle focused view / full diagram (V)" aria-keyshortcuts="v">Full diagram</button>
          <button id="deck-zoom" title="Enlarge the slide">Zoom</button>
          <button id="deck-theme" aria-label="Switch theme">◐</button>
          <button id="deck-fullscreen">Full screen</button>
        </div>
      </header>
      <div class="deck-viewport" id="deck-viewport">
        <section class="deck-canvas" id="deck-canvas" aria-label="Current slide">
          <header class="deck-slide-header"><div id="deck-brand"></div><span id="deck-section"></span></header>
          <h1 id="deck-title" class="deck-title"></h1>
          <div class="deck-setup" id="deck-setup"></div>
          <div class="deck-visual" id="deck-visual"></div>
          <p class="deck-summary" id="deck-summary" aria-live="polite"></p>
          <p class="deck-scope" id="deck-scope"></p>
          <footer class="deck-slide-footer"><span id="deck-copyright"></span><a id="deck-reference" href="#content">Article section</a><span id="deck-page" class="deck-page"></span></footer>
        </section>
      </div>
      <nav class="deck-navigation" aria-label="Slide and diagram navigation">
        <button id="deck-back" title="Previous step (Left arrow)">← Previous</button>
        <span id="deck-step" aria-live="polite"></span>
        <span class="deck-shortcuts">Arrow keys step through figures · Shift + arrow changes slide · V toggles diagram view · N opens notes</span>
        <button id="deck-next" title="Next step (Right arrow)">Next →</button>
      </nav>
      <aside class="deck-drawer" id="deck-controls-panel" aria-label="Figure controls" hidden><div class="deck-drawer-heading"><h2>Figure controls</h2><button data-close="controls" aria-label="Close controls">×</button></div><div id="deck-controls-content"></div></aside>
      <aside class="deck-drawer" id="deck-notes-panel" aria-label="Speaker notes" hidden><div class="deck-drawer-heading"><h2>Speaker notes</h2><button data-close="notes" aria-label="Close notes">×</button></div><div id="deck-notes-content"></div></aside>
      <div class="deck-toast" id="deck-toast" role="status" hidden></div>`;
    document.body.append(shell);
    canvas = $('deck-canvas');
    const brand = cleanClone(original.brand);
    brand.removeAttribute('href');
    $('deck-brand').append(brand);
    $('deck-copyright').textContent = original.copyright.textContent;
    $('deck-slide-select').innerHTML = scenes.map((s, i) => `<option value="${i}">${String(i + 1).padStart(2, '0')} · ${Views.esc(titleFor(s))}</option>`).join('');
    $('deck-slide-select').addEventListener('change', e => show(Number(e.target.value)));
    $('deck-article').addEventListener('click', exit);
    $('deck-reference').addEventListener('click', e => { e.preventDefault(); exit(); });
    $('deck-back').addEventListener('click', () => step(-1));
    $('deck-next').addEventListener('click', () => step(1));
    $('deck-controls').addEventListener('click', () => drawer('controls', !controlsOpen));
    $('deck-notes').addEventListener('click', () => drawer('notes', !notesOpen));
    shell.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => drawer(b.dataset.close, false)));
    $('deck-fit').addEventListener('click', toggleView);
    $('deck-zoom').addEventListener('click', () => { zoomed = !zoomed; resize(); $('deck-zoom').textContent = zoomed ? 'Fit slide' : 'Zoom'; });
    $('deck-fullscreen').addEventListener('click', fullScreen);
    $('deck-theme').addEventListener('click',()=>{$('theme-toggle').click();updatePresenter();});
    $('deck-presenter').addEventListener('click',presenter);
    $('deck-export').addEventListener('click',exportPDF);
  }
  /** Open the complete stepped print view from the presentation controls. */
  function exportPDF() {
    if (!window.DeckPDF?.open()) toast('Allow pop-ups for this page, then select Export PDF again.');
  }
  function toggleView(){const scene=scenes[index];if(!scene.camera)return;figures[scene.figure]?.querySelectorAll('svg[id]').forEach(app.finishTransition);fullView=!fullView;update();}
  function drawer(which, open) {
    if (open) { const other = which === 'controls' ? 'notes' : 'controls'; $('deck-' + other + '-panel').hidden = true; $('deck-' + other).setAttribute('aria-expanded', 'false'); if(other === 'controls') controlsOpen=false; else notesOpen=false; }
    if (which === 'controls') controlsOpen = open; else notesOpen = open;
    const panel = $('deck-' + which + '-panel');
    panel.hidden = !open;
    $('deck-' + which).setAttribute('aria-expanded', String(open));
    if (open) panel.querySelector('button').focus();
  }
  function toast(message) {
    $('deck-toast').textContent = message; $('deck-toast').hidden = false;
    setTimeout(() => { if ($('deck-toast')) $('deck-toast').hidden = true; }, 2500);
  }
  async function fullScreen() {
    const target = document.documentElement;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (target.requestFullscreen) await target.requestFullscreen();
      else if (target.webkitRequestFullscreen) target.webkitRequestFullscreen();
      else toast('Use your browser’s full-screen command.');
    } catch { toast('Use your browser’s full-screen command.'); }
  }

  function presenter(){
    speakerWindow=window.open('about:blank','dist-moe-presenter','width=780,height=900');
    if(!speakerWindow){drawer('notes',true);return;}
    const css=new URL('styles.css',location.href).href;
    speakerWindow.document.open();
    speakerWindow.document.write(`<!doctype html><html><head><title>Dist-MoE speaker notes</title><link rel="stylesheet" href="${Views.esc(css)}"><style>body{padding:22px;font-size:16px;line-height:1.65}header{position:sticky;top:0;background:var(--paper);padding:10px 0;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}button{font:inherit;color:var(--ink);background:var(--raised);border:1px solid var(--line);border-radius:5px;padding:6px 14px}h1{font-size:24px;line-height:1.3;margin:22px 0 12px}#speaker-next{color:var(--faint);font-size:14px}main p{font-size:15px;max-width:none}main table{min-width:0;font-size:12px}main .phase-caption{margin:0;padding:12px;font-size:14px}main figcaption{padding:0;border:0;font-size:13px}main .support-table{max-width:none}main h3{font-size:19px}main code{font-size:12px}</style></head><body><header><button id="speaker-back">← Previous</button><span id="speaker-count"></span><button id="speaker-forward">Next →</button></header><h1 id="speaker-title"></h1><p id="speaker-next"></p><main id="speaker-notes"></main></body></html>`);
    speakerWindow.document.close();
    speakerWindow.document.getElementById('speaker-back').onclick=()=>step(-1);
    speakerWindow.document.getElementById('speaker-forward').onclick=()=>step(1);
    speakerWindow.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();step(1,e.shiftKey);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();step(-1,e.shiftKey);}});
    updatePresenter();
  }
  function updatePresenter(){
    if(!speakerWindow||speakerWindow.closed||!active)return;
    const doc=speakerWindow.document;
    doc.documentElement.classList.toggle('light',document.documentElement.classList.contains('light'));
    doc.getElementById('speaker-title').textContent=$('deck-title').textContent;
    doc.getElementById('speaker-count').textContent=`Slide ${index+1} / ${scenes.length} · ${$('deck-step').textContent}`;
    doc.getElementById('speaker-next').textContent=index+1<scenes.length?'Next slide: '+titleFor(scenes[index+1]):'End of presentation';
    const notes=doc.getElementById('speaker-notes');notes.replaceChildren(doc.importNode($('deck-notes-content'),true));
  }


  function checkpoints(scene){
    return scene.phases || (scene.actions?scene.actions.map(action=>Moe.schedule.indexOf(action)).filter(p=>p>=0):null);
  }

  function range(scene) {
    const c = app.controllers[scene.controller];
    const points=checkpoints(scene);
    if(points?.length)return [points[0],points[points.length-1]];
    return scene.range ? [scene.range[0], scene.range[1] ?? c.count - 1] : [0, 0];
  }
  function slideSteps(scene) {
    if(scene.builds)return scene.builds.map((_,index)=>index);
    const c=app.controllers[scene.controller];
    if(!c)return [0];
    const sparse=checkpoints(scene);if(sparse)return sparse;
    const [first,last]=range(scene);
    return Array.from({length:last-first+1},(_,i)=>first+i);
  }
  function step(direction, slideOnly = false) {
    const scene = scenes[index], c = app.controllers[scene.controller];
    if(scene.builds&&!slideOnly){const target=build+direction;if(target>=0&&target<scene.builds.length){setBuild(scene,target);update();return;}}
    if (c && !slideOnly) {
      const points=checkpoints(scene);
      if(points){const target=direction>0?points.find(p=>p>c.phase):[...points].reverse().find(p=>p<c.phase);if(target!=null){c.set(target);update();return;}}
      else{const [first,last]=range(scene);if(direction>0&&c.phase<last){c.set(c.phase+1);update();return;}if(direction<0&&c.phase>first){c.set(c.phase-1);update();return;}}
    }
    const next = index + direction;
    if (next >= 0 && next < scenes.length) show(next, direction < 0);
  }
  /** Apply a presentation build through the same controls used by the article. */
  function setBuild(scene, position) {
    build=position;
    source(scene.builds[position]).click();
  }
  function part(figure, id) {
    const element = figure.querySelector('#' + id);
    if (!element) return;
    let container = element;
    while (container.parentElement !== figure && container.parentElement) container = container.parentElement;
    container.classList.add('deck-part');
  }
  function clearReservation(node) {
    if(!node)return;
    node.classList.remove('deck-reserved');node.style.removeProperty('--deck-reserved-height');
  }
  function reserveLayout(scene) {
    // Measure the real shared content across builds and supported modes once.
    // Reserving those heights keeps SVG scale/position unchanged during playback.
    const figure=figures[scene.figure],c=app.controllers[scene.controller];
    const bands=['deck-title','deck-setup','deck-summary','deck-scope'].map($);
    const extras=figure?[...figure.children].filter(n=>n.classList.contains('deck-part')&&!n.matches('.diagram-scroll,.table-scroll')):[];
    const nodes=[...bands,...extras],heights=new Map();nodes.forEach(clearReservation);
    const roots=[figure,$('deck-controls-content')].filter(Boolean);
    const inputs=[...new Set(roots.flatMap(n=>[...n.querySelectorAll('input,select')]))].map(n=>({n,value:n.value,checked:n.checked}));
    const buttons=[...new Set(roots.flatMap(n=>[...n.querySelectorAll('[aria-pressed]')]))].map(n=>({n,value:n.getAttribute('aria-pressed')}));
    const savedState={...app.state},phase=c?.phase,count=c?.count,savedBuild=build;
    const collect=()=>{update();for(const n of nodes)if(n.getClientRects().length)heights.set(n,Math.max(heights.get(n)||0,n.scrollHeight,n.offsetHeight));};
    const pass=()=>{if(scene.builds){for(const p of slideSteps(scene)){setBuild(scene,p);collect();}}else if(c){for(const p of slideSteps(scene)){c.phase=p;c.render();collect();}}else collect();};
    const setValue=(id,value)=>{if($(id))$(id).value=String(value);};
    const restoreSource=()=>{
      Object.assign(app.state,savedState);
      build=savedBuild;
      inputs.forEach(({n,value,checked})=>{n.value=value;if(n.type==='checkbox')n.checked=checked;});
      buttons.forEach(({n,value})=>n.setAttribute('aria-pressed',value));
      if(c){c.phase=phase;c.count=count;c.render();}
      else if(scene.figure==='precision-figure')app.renderPrecision();
      else if(scene.figure==='memory-figure')app.renderMemory();
      else if(scene.figure==='recompute-figure')app.renderRetention();
    };
    calibrating=true;canvas.classList.add('deck-calibrating');c?.stop();
    try {
      app.withoutAnimation(()=>{
        pass();
        if(scene.figure==='precision-figure'){
          for(const format of ['bf16','mxfp8','nvfp4'])for(const direction of ['forward','backward'])for(const outlier of [false,true])for(const byte of [0,7,31]){
            app.state.format=format;app.state.direction=direction;app.state.quantByte=byte;$('quant-outlier').checked=outlier;app.renderPrecision();pass();
          }
        }else if(scene.chapter==='results'){
          window.ResultsApp.sample(scene.controller,pass);
        }else if(scene.controller==='packing'){
          for(const rank of [0,1])for(const budget of [4096,24576,35840]){app.state.packingRank=rank;app.state.packingBudget=budget;pass();}
        }else if(scene.figure==='recompute-figure'){
          for(const precision of ['bf16','mxfp8'])for(const mode of ['save','recompute']){
            app.state.retentionPrecision=precision;app.state.retention=mode;app.renderRetention();pass();
            if(scene.prepare==='address')for(const button of $('offset-table').querySelectorAll('[data-offset]')){app.state.offset=button.dataset.offset;app.renderRetention();pass();}
          }
        }else if(scene.controller==='fusion'){
          for(const pipeline of ['staged','mega'])for(const chunk of [0,1,2]){app.state.fusion=pipeline;app.state.fusionChunk=chunk;c.count=app.fusionCount(pipeline);pass();}
        }else if(scene.controller==='slots'&&scene.slotDisplay==='reuse'){
          for(const policy of ['stage','microbatch']){app.state.slotPolicy=policy;pass();}
        }else if(scene.controller==='kernel'){
          for(const [m,n,tail,k]of [[128,128,'full',0],[256,256,'tail',3]]){setValue('tile-M',m);setValue('tile-N',n);setValue('tile-tail',tail);setValue('tile-K',k);pass();}
        }else if(scene.controller==='tokens'){
          for(const token of ['all','A','B','C','D']){app.state.token=token;pass();}
        }else if(scene.figure==='memory-figure'&&app.renderMemory()){
          const originalShape=Object.fromEntries(['T','D','F','L','N'].map(k=>[k,$('config-'+k).value]));
          for(const shape of [originalShape,{T:8192,D:4096,F:14336,L:64,N:8}]){
            for(const [key,value]of Object.entries(shape))setValue('config-'+key,value);
            app.state.memorySlotPolicy='factor';setValue('config-device-factor',1.5);setValue('config-total-factor',16);
            for(const precision of ['bf16','mxfp8'])for(const pipeline of ['staged','mega'])for(const vmm of [false,true])for(const factor of [0,1]){
              app.state.memoryPrecision=precision;app.state.memorySlotFactor=factor;setValue('config-pipeline',pipeline);$('vmm-enabled').checked=vmm;
              for(const calculation of ['saved','device','total']){app.state.calculation=calculation;app.renderMemory();pass();}
            }
          }
        }
        restoreSource();collect();
      });
    } finally {
      // Also restore if a future source renderer rejects one of the samples.
      app.withoutAnimation(restoreSource);canvas.classList.remove('deck-calibrating');calibrating=false;
    }
    for(const [node,height]of heights){node.style.setProperty('--deck-reserved-height',Math.ceil(height+1)+'px');node.classList.add('deck-reserved');}
    if(c)c.set(phase);
  }
  function show(i, lastStep = false, url = true) {
    if (!active) return;
    clearTimeout(slideFadeTimer); slideFade?.cancel();
    restore();
    index = Math.max(0, Math.min(scenes.length - 1, i));
    const scene = scenes[index];
    build=0;
    fullView = false;
    ['deck-title','deck-setup','deck-summary','deck-scope'].forEach(id=>clearReservation($(id)));
    $('deck-visual').replaceChildren(); $('deck-controls-content').replaceChildren();
    $('deck-summary').textContent = ''; $('deck-setup').textContent = '';
    canvas.dataset.kind = scene.kind || scene.id;
    canvas.classList.toggle('deck-table-slide', !!scene.table);
    $('deck-slide-select').value = index;
    if (scene.figure) {
      const figure = figures[scene.figure];
      const c = app.controllers[scene.controller];
      move(figure, $('deck-visual'));
      const parts = scene.parts || [c?.svg().id];
      parts.filter(Boolean).forEach(id => part(figure, id));
      const controls = [...figure.children].filter(node => {
        if (!node.matches('.figure-controls,.phase-tabs,.config-fold,.slot-budget-panel,.calculation-details')) return false;
        if (['packing-display','slots-display','retention-display','slot-plan-details'].includes(node.id)) return false;
        if (node.querySelector('button[data-phase]')) return false;
        if (scene.slotDisplay==='liveness' && node.id==='slot-policy-controls') return false;
        if (scene.id==='allocation' && node.id==='calculation-tabs') return false;
        if (node.id==='memory-examples') return false;
        if (scene.prepare==='lifetime' && node.querySelector('#offset-table')) return false;
        return true;
      });
      controls.forEach(n => move(n, $('deck-controls-content')));
      figure.querySelectorAll('svg[id]').forEach(svg => { if (!savedCamera.has(svg)) savedCamera.set(svg, svg.getAttribute('viewBox')); });
      if(scene.slotDisplay)document.querySelector(`#slots-display [data-view="${scene.slotDisplay}"]`).click();
      if(scene.display)document.querySelector(`#packing-display [data-view="${scene.display}"]`).click();
      if (scene.prepare) document.querySelector(`#retention-display [data-view="${scene.prepare}"]`).click();
      if(scene.preset)source(scene.preset).click();
      if(scene.builds)setBuild(scene,lastStep?scene.builds.length-1:0);
      if (c) c.set(lastStep ? range(scene)[1] : range(scene)[0]);
      figure.querySelectorAll('svg[id]').forEach(app.finishTransition);
      if (scene.id === 'moe') {
        move(original.mlp, $('deck-visual'));
        original.mlp.classList.add('deck-part');
      }

    } else if (scene.kind === 'cover') {
      const box = document.createElement('div'); box.className = 'deck-cover';
      [original.title, original.tagline, original.intro].forEach(node => box.append(cleanClone(node)));
      const author = cleanClone(original.author);
      const names = (original.author.querySelector('strong')?.textContent || original.author.textContent.replace(/^Written by\s+/i, '')).split(',').map(name => name.trim());
      author.replaceChildren();
      names.forEach((name, i) => {
        if (i) {
          const separator = document.createElement('span'); separator.className = 'deck-author-separator'; separator.textContent = ' · ';
          author.append(separator);
        }
        const credit = document.createElement('span'); credit.className = 'deck-author-name'; credit.textContent = name; author.append(credit);
      });
      box.append(author);
      $('deck-visual').append(box);
    } else if (scene.kind === 'orientation') {
      move(original.orientation, $('deck-visual'));
    } else if (scene.kind === 'resources') {
      const box = document.createElement('div'); box.className = 'deck-static deck-resources';
      box.append(cleanClone(original.support), cleanClone(original.resources)); $('deck-visual').append(box);
    } else if (scene.kind === 'code') {
      move(original.code, $('deck-controls-content'));
      const box = document.createElement('div'); box.className = 'deck-code';
      box.innerHTML = '<div class="deck-code-label" id="deck-code-label">Quickstart excerpt</div><pre><code id="deck-code-excerpt"></code></pre>';
      $('deck-visual').append(box);

    }
    updateNotes(scene);
    update(); reserveLayout(scene); update(); resize();
    const observed=scene.figure?figures[scene.figure]:scene.kind==='code'?original.code:null;
    if(observed){observer=new MutationObserver(queueUpdate);observer.observe(observed,{childList:true,characterData:true,subtree:true});}
    if (url) setURL(true);
    if (!exporting && !reduced.matches) { slideFade=canvas.animate([{ opacity: .35 }, { opacity: 1 }], { duration: 220, easing: 'ease-out' }); slideFadeTimer=setTimeout(()=>slideFade?.cancel(),260); }
  }
  function queueUpdate() {
    if (updateQueued || !active || exporting) return;
    updateQueued = true;
    queueMicrotask(() => { updateQueued = false; if (active) update(); });
  }
  function cameraFor(scene) {
    // One framing window per slide. Steps change content, never the camera.
    return fullView || !Array.isArray(scene.camera) ? null : scene.camera;
  }
  function applyCamera(scene) {
    if (!scene.figure) return;
    const figure = figures[scene.figure];
    const visible = [...figure.querySelectorAll('.deck-part svg[id]:not(.transition-overlay)')];
    const camera = cameraFor(scene);
    for (const svg of visible) {
      const saved = savedCamera.get(svg);
      const originalBounds=scene.bounds||(saved || svg.getAttribute('viewBox')).split(/\s+/).map(Number);
      let bounds=originalBounds;
      if(camera){const [x,y,w,h]=camera,[ox,oy,ow,oh]=originalBounds,padx=scene.pane?0:Math.max(85,w*.18),pady=scene.pane?0:Math.max(45,h*.18),left=Math.max(ox,x-padx-2),top=Math.max(oy,y-pady-2);bounds=[left,top,Math.min(ox+ow,x+w+padx+2)-left,Math.min(oy+oh,y+h+pady+2)-top];}
      svg.setAttribute('viewBox', bounds.join(' '));
      const parent=svg.parentElement,ratio=bounds[2]/bounds[3],width=Math.min(parent.clientWidth,parent.clientHeight*ratio);
      svg.style.setProperty('--deck-svg-width',width+'px');svg.style.setProperty('--deck-svg-height',width/ratio+'px');
      svg.querySelectorAll('.deck-camera-clipped').forEach(n => n.classList.remove('deck-camera-clipped'));
      const [x, y, w, h] = bounds;
      const outside = b => b.x < x - .1 || b.y < y - .1 || b.x + b.width > x + w + .1 || b.y + b.height > y + h + .1;
      if (camera) {
        svg.querySelectorAll('.edge-payload-label').forEach(label => { if (outside(label.getBBox())) label.classList.add('deck-camera-clipped'); });
        svg.querySelectorAll('text').forEach(text => { if(!text.closest('[data-sticky-slot]')){const b=text.getBBox();const ownerMargin=svg.id==='slots-svg'&&text.hasAttribute('data-owner-label')&&b.x+b.width*1.25+8>x+w;if(outside(b)||ownerMargin||(svg.id==='slots-svg'&&x>0&&b.x<x+129))text.classList.add('deck-camera-clipped');} });
      }
      if (svg.id === 'slots-svg') svg.querySelector('[data-sticky-slot]')?.setAttribute('transform', `translate(${x},0)`);
      svg.classList.remove('deck-map');
      window.Views.positionMath(svg);
    }
    $('deck-fit').hidden = !scene.camera;
    $('deck-fit').textContent = fullView ? 'Focused view · V' : 'Full diagram · V';
    $('deck-fit').setAttribute('aria-pressed',String(fullView));
  }
  function codeBlock(lines, needle) {
    const start = lines.findIndex(line => line.includes(needle));
    if (start < 0) return [];
    let depth = 0, result = [], began = false;
    for (let i = start; i < lines.length; i++) {
      const line = lines[i]; result.push(line);
      for (const char of line.replace(/#.*$/, '')) { if (char === '(') { depth++; began = true; } else if (char === ')') depth--; }
      if (began && depth === 0) break;
    }
    const indent = result[0].match(/^\s*/)[0].length;
    return result.map(line => line.slice(indent));
  }
  function updateCode(scene) {
    const lines = [...$('quickstart-code').querySelectorAll('.code-line')].map(n => n.textContent);
    let excerpt;
    if (scene.excerpt === 'config') {
      const dims = lines.find(line => /^T, D, F, K, E_local/.test(line));
      const config = codeBlock(lines, 'dist_moe.Config(');
      const reference = config.map(line => line.match(/\bblock_scaled\s*=\s*([A-Za-z_]\w*)\s*,/)).find(Boolean)?.[1];
      const definition = reference && lines.find(line => line.split('=')[0].trim() === reference && line.includes('dist_moe.BlockScaledConfig('));
      const policy = definition ? codeBlock(lines, definition.trim()) : [];
      excerpt = [...(dims ? [dims, ''] : []), ...policy, ...(policy.length ? [''] : []), ...config];
    } else {
      const first = lines.findIndex(line => line.includes('dist_moe.prepare_block_scaled_weight('));
      const end = lines.findIndex((line, i) => i > first && line.trim() === 'try:');
      const preparation = first >= 0 && end > first ? lines.slice(Math.max(0, first - 2), end) : [];
      excerpt = [...codeBlock(lines, 'dist_moe.create_context('), '', ...preparation, ...(preparation.length ? [''] : []), ...codeBlock(lines, 'dist_moe.routed_experts(')];
      if (!excerpt.some(line => line.includes('routed_experts'))) excerpt = lines.filter(line => line.includes('dist_moe.plan_memory') || line.includes('print(plan'));
    }
    const cut=excerpt.length>22?Math.ceil(excerpt.length/2):excerpt.length;
    const columns=[excerpt.slice(0,cut),...(cut<excerpt.length?[excerpt.slice(cut)]:[])];
    $('deck-code-excerpt').classList.toggle('deck-code-columns',columns.length>1);
    $('deck-code-excerpt').innerHTML=columns.map((lines,c)=>`<span class="deck-code-column">${lines.map((line,i)=>`<span class="deck-code-line" data-source-line="${c*cut+i+1}">${app.highlight(line)||' '}</span>`).join('')}</span>`).join('');
    $('deck-visual').dataset.codeLines = excerpt.length;
    $('deck-code-excerpt').style.fontSize = excerpt.length > 24 ? '24px' : '30px';
    $('deck-code-label').textContent = $('code-tabs').querySelector('[aria-pressed="true"]').textContent + ' · excerpt';
  }

  function firstSentenceFragment(node,count=2){
    const clone=node.cloneNode(true);clone.querySelector('strong')?.remove();
    const text=clone.textContent,start=text.length-text.trimStart().length,end=start+sentence(text.trim(),count).length;
    const walker=document.createTreeWalker(clone,NodeFilter.SHOW_TEXT);let n,offset=0,startNode,endNode,startOffset=0,endOffset=0;
    while(n=walker.nextNode()){
      if(!startNode&&offset+n.length>start){startNode=n;startOffset=start-offset;}
      if(offset+n.length>=end){endNode=n;endOffset=end-offset;break;}offset+=n.length;
    }
    if(!startNode||!endNode)return document.createTextNode(sentence(text.trim(),count));
    const range=document.createRange();range.setStart(startNode,startOffset);range.setEnd(endNode,endOffset);return range.cloneContents();
  }
  function scopeFor(scene){
    const f=figures[scene.figure];if(!f)return '';
    if(scene.id==='results-metrics')return 'Extra padding/replay FLOPs are not separately instrumented. All routed-operation costs remain in measured latency; effective HBM bandwidth is a model.';
    if(scene.chapter==='results')return window.ResultsApp.scope(scene.controller,app.controllers[scene.controller].phase);
    if(scene.id==='capabilities')return source('#capability-scope').textContent;
    const text=f.querySelector('figcaption').textContent;
    let sentences;
    try{sentences=[...new Intl.Segmenter('en',{granularity:'sentence'}).segment(text)].map(s=>s.segment.trim());}catch{sentences=[text];}
    if(scene.controller==='slots')return sentences.filter(s=>/continuation/.test(s)).join(' ');
    if(scene.id==='lifetimes')return source('#dependency-math-note').textContent.match(/These are ideal dependency equations:.*$/)?.[0]||'';
    if(scene.figure==='precision-figure'){
      const chosen=sentences.filter(s=>app.state.format==='nvfp4'?/NVFP4/.test(s):/finite MXFP8/.test(s)).join(' ');
      const weight=[...f.querySelectorAll('#intermediate-table tbody tr')].find(tr=>tr.firstElementChild.textContent==='W13 / W2 weights');
      return chosen+(app.state.format==='nvfp4'&&weight?' '+weight.lastElementChild.textContent+'.':'');
    }
    if(scene.controller==='kernel')return sentences.filter(s=>/Matrix geometry|Gray rows/.test(s)).join(' ');
    if(scene.figure==='memory-figure')return sentences.find(s=>/These logical bytes/.test(s))||'';
    if(scene.controller==='fusion')return sentences[0]||'';
    return sentences.find(s=>/illustrative|not measured/.test(s))||'';
  }

  function update() {
    if (!active) return;
    const scene = scenes[index], figure = figures[scene.figure];
    $('deck-title').textContent = titleFor(scene, true);
    $('deck-section').textContent = figure?.querySelector('.fig-number')?.textContent || source('#' + scene.chapter)?.dataset.title || '';
    $('deck-page').textContent = `${index + 1} / ${scenes.length}`;
    $('deck-reference').href = '#' + scene.chapter;
    const c = app.controllers[scene.controller];
    const steps=slideSteps(scene),at=scene.builds?build:c?steps.indexOf(c.phase):0;
    const position=at>=0?at:Math.max(0,steps.findIndex(p=>p>=c.phase));
    $('deck-step').textContent=`Step ${position+1} / ${steps.length}${scene.controller==='slots'?' · '+Moe.schedule[c.phase]:''}`;
    $('deck-back').disabled = index === 0;
    $('deck-next').disabled = index === scenes.length - 1;
    $('deck-controls').hidden = !$('deck-controls-content').children.length;
    $('deck-fit').hidden = !scene.figure || !scene.camera;
    shell.querySelector('.deck-shortcuts').textContent='Arrow keys step through figures · Shift + arrow changes slide'+(scene.camera?' · V toggles focused/full view':'')+' · N opens notes';
    if (scene.setup) {
      const node = figure?.querySelector(scene.setup) || source(scene.setup);
      $('deck-setup').textContent = node?.textContent || '';
    }
    const caption=captionNode(scene),summary=source(`[data-slide-summary="${scene.id}"]`);
    const dynamic=caption&&(scene.controller||scene.builds||scene.figure==='recompute-figure'&&scene.id!=='lifetimes');
    const topic=scene.controller==='packing'?caption?.querySelector(scene.id==='layer-packing'?'.packing-step-summary':'.decision-step-summary'):null;
    let takeaway=topic?cleanClone(topic):dynamic?firstSentenceFragment(caption,scene.controller==='slots'?3:2):summary?cleanClone(summary):null;
    if(scene.id==='layer-packing'&&topic){takeaway=document.createDocumentFragment();takeaway.append(firstSentenceFragment(caption.querySelector('.decision-step-summary'),1),document.createTextNode(' '),firstSentenceFragment(topic,1));}
    $('deck-summary').replaceChildren(...(takeaway?[...takeaway.childNodes]:[]));
    $('deck-scope').textContent=scopeFor(scene);
    if (scene.kind === 'code') updateCode(scene);
    if(calibrating)return;
    applyCamera(scene);
    $('deck-notes-caption')?.replaceChildren(...(caption ? [cleanClone(caption)] : []));
    const evidence=$('deck-notes-evidence');if(evidence){evidence.replaceChildren();const ids=scene.id==='precision'?['intermediate-table']:scene.id==='memory-costs'?['calculation-table']:scene.figure==='recompute-figure'?['offset-table']:scene.figure==='packing-figure'?['packing-table']:[];if(scene.chapter==='results')ids.push(scene.id==='results-metrics'?'metrics-worked-example':scene.controller+'-table');ids.forEach(id=>{const table=$(id);if(table)evidence.append(cleanClone(table));});}
    updatePresenter();

  }
  function updateNotes(scene) {
    const box = $('deck-notes-content'); box.replaceChildren();
    const heading = document.createElement('h3'); heading.textContent = titleFor(scene); box.append(heading);
    const caption = document.createElement('div'); caption.id = 'deck-notes-caption'; box.append(caption);
    const evidence=document.createElement('div');evidence.id='deck-notes-evidence';box.append(evidence);
    const figure = figures[scene.figure];
    if (figure) box.append(cleanClone(figure.querySelector('figcaption')));
    (scene.id==='moe'?[...source('#moe-primer').children].filter(n=>n.tagName==='P'):(paragraphs[scene.chapter] || [original.intro])).forEach(node => box.append(cleanClone(node)));
    if (scene.kind === 'code') box.append(cleanClone(original.support));
  }
  function resize() {
    if (!active) return;
    const view = $('deck-viewport'), width = view.clientWidth - 24, height = view.clientHeight - 20;
    const scale = Math.max(.1, Math.min(width / 1920, height / 1080)) * (zoomed ? (view.clientWidth<750?3.5:1.65) : 1);
    canvas.style.transform = `scale(${scale})`;
    canvas.style.left = Math.max(12, (view.clientWidth - 1920 * scale) / 2) + 'px';
    canvas.style.top = Math.max(10, (view.clientHeight - 1080 * scale) / 2) + 'px';
    view.classList.toggle('deck-zoomed', zoomed);
    applyCamera(scenes[index]);
  }
  function setURL(slides, push = false) {
    const url = new URL(location.href);
    if (slides) { url.searchParams.set('view', 'slides'); url.hash = 'slide-' + scenes[index].id; }
    else { url.searchParams.delete('view'); url.hash = scenes[index].chapter; }
    history[push ? 'pushState' : 'replaceState']({}, '', url);
  }
  /** Capture the curated builds with the same layout, captions, and renderers. */
  function printFrames() {
    if (!printMode || !active) throw new Error('Open the dedicated PDF print view first.');
    const frames = [];
    exporting = true;
    try {
      app.withoutAnimation(() => {
        scenes.forEach((scene, slide) => {
          show(slide, false, false);
          const phases = slideSteps(scene), controller = app.controllers[scene.controller];
          phases.forEach((phase, step) => {
            if(scene.builds)setBuild(scene,phase);
            controller?.set(phase);
            update();
            frames.push({ node: window.DeckPDF.freeze(canvas), scene: scene.id, slide, step, steps: phases.length, phase });
          });
        });
      });
      return frames;
    } finally {
      restore();
      active = false;
      shell.remove(); shell = null; canvas = null;
      document.body.classList.remove('presentation-mode');
      exporting = false;
    }
  }
  function enter(push = false) {
    if (active) return;
    savedFocus = document.activeElement;
    active = true;
    Object.values(app.controllers).forEach(c => c.stop());
    document.querySelectorAll('svg[id]').forEach(app.finishTransition);
    article.classList.add('deck-source'); article.inert = true; article.setAttribute('aria-hidden', 'true');
    document.body.classList.add('presentation-mode');
    createShell();
    const requested = location.hash.replace('#slide-', '');
    const i = sceneIndex(requested);
    show(i >= 0 ? i : 0, false, false);
    if (!printMode) { setURL(true, push); $('deck-next').focus(); }
  }
  function exit() {
    if (!active) return;
    const chapter = scenes[index].chapter;
    restore(); active = false;
    if(document.fullscreenElement)document.exitFullscreen?.().catch(()=>{});else if(document.webkitFullscreenElement)document.webkitExitFullscreen?.();
    if(speakerWindow&&!speakerWindow.closed)speakerWindow.close();speakerWindow=null;
    shell.remove(); shell = null; canvas = null;
    document.body.classList.remove('presentation-mode');
    article.classList.remove('deck-source'); article.inert = false; article.removeAttribute('aria-hidden');
    setURL(false);
    requestAnimationFrame(() => { $('' + chapter)?.scrollIntoView({ behavior: 'instant', block: 'start' }); (savedFocus || $('present-link')).focus(); });
  }
  $('present-link').addEventListener('click', e => { e.preventDefault(); enter(true); });
  document.addEventListener('keydown', e => {
    if (active && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'p') { e.preventDefault(); exportPDF(); return; }
    if (!active || e.altKey || e.metaKey || e.ctrlKey) return;
    if (e.key === 'Escape') { drawer('controls', false); drawer('notes', false); if (zoomed) { zoomed = false; resize(); } return; }
    if (e.target.closest?.('input,select,textarea,[contenteditable="true"]')) return;
    if (e.key.toLowerCase() === 'n') { drawer('notes',!notesOpen); return; }
    if (e.key.toLowerCase() === 'c') { drawer('controls',!controlsOpen); return; }
    if (e.key.toLowerCase() === 'v') { e.preventDefault();toggleView();return; }
    if (controlsOpen || notesOpen) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); step(1, e.shiftKey); }
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); step(-1, e.shiftKey); }
    else if (e.key === 'Home') { e.preventDefault(); show(0); }
    else if (e.key === 'End') { e.preventDefault(); show(scenes.length - 1); }
        else if (e.key.toLowerCase() === 'f') fullScreen();
    else if (e.key.toLowerCase() === 't') { $('theme-toggle').click();updatePresenter(); }
  });
  window.addEventListener('resize', resize);
  document.addEventListener('fullscreenchange', resize);
  window.addEventListener('popstate', () => {
    if (new URLSearchParams(location.search).get('view') === 'slides') {
      if (!active) enter();
      else { const i = sceneIndex(location.hash.replace('#slide-','')); if (i >= 0) show(i, false, false); }
    } else if (active) exit();
  });
  window.Presentation = { enter, exit, show, step, resize, printFrames, scenes, get active() { return active; }, get index() { return index; } };
  if (printMode || new URLSearchParams(location.search).get('view') === 'slides') enter();
})();
