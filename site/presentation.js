'use strict';
// Presentation is a view of index.html. Content and figures stay in that file.
(() => {
  const $ = id => document.getElementById(id);
  const article = $('content');
  const app = window.Explainer;
  if (!article || !app) return;

  const source = selector => article.querySelector(selector) || document.querySelector(selector);
  const original = {
    title: source('h1'), tagline: source('.deck'), intro: source('.intro'), documentTitle: document.title,
    author: source('.author-byline'), brand: document.querySelector('.wordmark'),
    repository: source('footer a[href*="github.com"]'),
    copyright: source('.copyright'), outline: source('.reading-map'),
    mlp: source('#precision .equation'),
    orientation: source('#orientation-panel'), resources: source('.source-grid'),
    support: source('.support-table'), code: source('.code-block')
  };
  const figures = Object.fromEntries([...article.querySelectorAll('figure[id]')].map(f => [f.id, f]));
  const paragraphs = Object.fromEntries([...article.querySelectorAll('section[id]')].map(section => [section.id, [...section.children].filter(n => ['P', 'DETAILS'].includes(n.tagName))]));
  paragraphs.motivation = [...source('#motivation').children].filter(node => node.tagName === 'P');
  paragraphs['mok-comparison'] = [...source('#mok-comparison').children].filter(node => node.tagName === 'P');

  // Topic titles and pacing belong to the presentation. Figures, explanations,
  // numerical values, and example programs use the shared article source.
  const mainScenes = [
    { id: 'title', kind: 'cover', chapter: 'content' },
    { id: 'moe', title: 'MoE Overview', figure: 'moe-figure', chapter: 'content', parts: ['moe-svg'] },
    { id: 'system', title: 'Distributed MoE: End-to-End Execution', figure: 'overview-figure', controller: 'overview', chapter: 'content', range: [0, 8], setup: '.diagram-key' },
    { id: 'capabilities', title: 'MoE Capabilities and Supported Modes', buildTitles: ['Open Source MoE Capability Comparison','Dist-MoE Supported Modes'], capabilitySteps: true, builds: ['capabilities','supported-modes'], figure: 'capability-figure', chapter: 'motivation', parts: ['capability-table', 'capability-notes', 'supported-modes'] },
    { id: 'comparison-policy', title: 'MoK vs Dist-MoE: Imbalanced Multi-Layer Workload', figure: 'compare-policy-figure', chapter: 'mok-comparison', parts: ['compare-policy-view'], setup: '#compare-policy-config' },
    { id: 'comparison-packing', title: 'MoK vs Dist-MoE: Multi-Layer Training Performance', figure: 'compare-packing-figure', chapter: 'mok-comparison', parts: ['compare-packing-legend','compare-packing-view'], setup: '#compare-packing-config' },
    { id: 'comparison-scaling', comparisonSteps: true, builds: ['#compare-scaling-steps [data-ep="16"]','#compare-scaling-steps [data-ep="64"]'], title: 'MoK vs Dist-MoE: Routing Imbalance', figure: 'compare-scaling-figure', chapter: 'mok-comparison', parts: ['compare-scaling-legend','compare-scaling-view'], setup: '#compare-scaling-config' },
    { id: 'comparison-width', comparisonSteps: true, builds: ['#compare-width-steps [data-ep="16"]','#compare-width-steps [data-ep="64"]'], title: 'MoK vs Dist-MoE: Expert Width', figure: 'compare-width-figure', chapter: 'mok-comparison', parts: ['compare-width-legend','compare-width-view'], setup: '#compare-width-config' },
    { id: 'routing', title: 'Routing Metadata and Symmetric Buffers', figure: 'token-figure', controller: 'tokens', chapter: 'tokens', range: [0, 6], bounds: [0,0,1120,740], setup: '#token-config' },
    { id: 'quantization', title: 'Block-Scaled Quantization', figure: 'precision-figure', chapter: 'precision', parts: ['precision-svg', 'quant-detail'] },
    { id: 'quantization-axis', title: 'Quantization Axes for Forward and Backward', kind: 'orientation', chapter: 'precision' },
    { id: 'precision', title: 'Intermediate Tensor Precision', figure: 'precision-figure', chapter: 'precision', parts: ['intermediate-table'], table: true },
    { id: 'memory-costs', title: 'Deriving a Layer’s Memory Cost', figure: 'memory-figure', chapter: 'memory', parts: ['memory-svg'], setup: '#memory-cost-shape', builds: ['#calculation-tabs [data-calculation="forward"]','#calculation-tabs [data-calculation="device"]'] },
    { id: 'allocation', title: 'Activation and Scratch Memory Planning', figure: 'memory-figure', chapter: 'memory', parts: ['memory-policy-summary','consumption-svg', 'memory-stats'], setup: '#memory-shape', builds: ['#memory-examples [data-memory-example="saved"]','#memory-examples [data-memory-example="scratch"]','#memory-examples [data-memory-example="fixed"]'] },
    { id: 'lifetimes', title: 'Forward and Backward Dependencies', figure: 'recompute-figure', chapter: 'recompute', parts: ['dependency-equations'], prepare: 'lifetime', setup: '.config-summary' },
    { id: 'layer-packing', title: 'Activation Packing and Dynamic Recompute', figure: 'packing-figure', controller: 'packing', chapter: 'recompute', range: [0, 8], parts: ['packing-svg'], display: 'packing', setup: '#packing-brief' },
    { id: 'gpu-tile', title: 'GPU Tile Movement and Compute', figure: 'kernel-figure', controller: 'kernel', chapter: 'kernels', range: [0, 5], setup: '#tile-address' },
    { id: 'mega-pipeline', title: 'MXFP8 Mega Forward', figure: 'fusion-figure', controller: 'fusion', fusionDirection:'forward', chapter: 'kernels', range: [0, null], bounds: [0,0,1600,550], setup: '#fusion-config' },
    { id: 'mega-backward', title: 'MXFP8 Mega Backward', figure: 'fusion-figure', controller: 'fusion', fusionDirection:'backward', chapter: 'kernels', range: [0,8], bounds: [0,0,1600,550], setup: '#fusion-config' },
    { id: 'pipeline-prefix', title: 'Pipeline Liveness and Slot Size', figure: 'slots-figure', controller: 'slots', chapter: 'slots', range: [0, 31], actions: ['0F0','2B0','0F4','0B7'], slotDisplay: 'liveness', parts: ['liveness-svg'], setup: '#slots-config' },
    { id: 'slot-reuse', title: 'Activation Slot Assignment and Reuse', figure: 'slots-figure', controller: 'slots', chapter: 'slots', range: [2, 6], actions: ['2F0','2B0','0F3'], slotDisplay: 'reuse', camera: [0, 0, 1146, 406], pane: true, parts: ['slots-svg','slot-selection'] },
    { id: 'results-metrics', title: 'Performance Metrics and Theoretical Peaks', figure: 'results-metrics-figure', chapter: 'results', parts: ['metrics-overview'], setup: '#metrics-setup' },
    { id: 'results-inference', title: 'Standalone Inference Performance', figure: 'results-efficiency-figure', controller: 'efficiency', chapter: 'results', range: [0,0], parts: ['efficiency-deck-view'], report: 'inference', preset: '#result-presets [data-preset="inference"]', setup: '#efficiency-config' },
    { id: 'results-training', title: 'Standalone Training Performance', figure: 'results-efficiency-figure', controller: 'efficiency', chapter: 'results', range: [0,0], parts: ['efficiency-deck-view'], report: 'training', preset: '#result-presets [data-preset="training"]', setup: '#efficiency-config' },
    { id: 'results-skew', title: 'Routing Imbalance: Work Growth and Execution Cost', figure: 'results-skew-figure', controller: 'skew', chapter: 'results', range: [0,0], parts: ['skew-deck-view'], report: 'skew', setup: '#skew-config' },
    { id: 'results-width', title: 'Expert Width: Useful Compute and Memory Traffic', figure: 'results-width-figure', controller: 'width', chapter: 'results', range: [0,0], parts: ['width-deck-view'], report: 'width', setup: '#width-config' },
    { id: 'public-api', title: 'Public API', kind: 'code', chapter: 'start', builds: ['#api-steps [data-api-step="0"]','#api-steps [data-api-step="1"]','#api-steps [data-api-step="2"]'] },
    { id: 'reading', title: 'Resources', kind: 'resources', chapter: 'start' }
  ];

  // Imported slides reference main scenes; short-talk memory views share their models.
  const lightningOrder=['moe','capabilities','comparison-policy','comparison-packing','comparison-scaling','comparison-width','public-api','system','mega-pipeline','mega-backward','results-metrics','results-inference','results-training','results-skew','results-width','reading'];
  const lightningScenes=[{id:'lightning-title',kind:'artwork-cover',chapter:'content',title:'Dist-MoE · PyTorch Conference NA 2026',alt:'Dist-MoE: Fused Comm-Compute Kernels and Activation Packing for Speed-of-Light MoEs. Shikai Li and Sanket Purandare, Meta. PyTorch Conference North America 2026.',asset:'assets/lightning-title.svg'},...lightningOrder.map(id=>mainScenes.find(scene=>scene.id===id))];
  lightningScenes.splice(11,0,
    {id:'lightning-memory',title:'Activation, Scratch, and VMM',figure:'memory-figure',chapter:'memory',parts:['consumption-svg'],compact:'tiers',bounds:[0,0,1040,510],memoryExample:{slotFactor:.35,deviceFactor:1.5,totalFactor:2},preset:'#memory-precision [data-precision="bf16"]',subtitle:'Saved-state capacity selects replay; scratch capacity bounds execution',summary:'The activation-slot factor controls optional saved bytes across layers. Scratch factors independently reserve capacity for per-call routing imbalance.'},
    {...mainScenes.find(scene=>scene.id==='memory-costs'),id:'lightning-working-set',title:'Forward and Backward Memory',compact:'working-set',bounds:[0,0,1040,525],preset:'#memory-precision [data-precision="bf16"]',setup:null},
    {...mainScenes.find(scene=>scene.id==='layer-packing'),id:'lightning-packing',compact:'packing',bounds:[0,0,1040,565],setup:null}
  );
  const decks={main:{id:'main',name:'Full Presentation',scenes:mainScenes},lightning:{id:'lightning',name:'Lightning',scenes:lightningScenes}};
  const deck=decks[new URLSearchParams(location.search).get('deck')]||decks.main;
  const scenes=deck.scenes;

  const sceneAliases = {'comparison-inference':'comparison-scaling','comparison-training':'comparison-scaling','comparison-graphs':'comparison-scaling',configuration:'public-api','expert-call':'public-api',assignments:'routing','expert-rows':'routing','return-slots':'routing','results-memory':'layer-packing','expert-math':'moe','dynamic-recompute':'layer-packing',offsets:'layer-packing','gather-tiles':'gpu-tile','tensor-cores':'gpu-tile','warp-groups':'mega-pipeline','chunk-readiness':'mega-pipeline'};
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
    const briefSVGs=mounted.flatMap(({node})=>[...node.querySelectorAll('svg[data-slide-detail="brief"]')]);
    briefSVGs.forEach(svg=>delete svg.dataset.slideDetail);
    app.withoutAnimation(()=>{
      if(briefSVGs.some(svg=>svg.id==='memory-svg'||svg.id==='consumption-svg'))app.renderMemory();
      if(briefSVGs.some(svg=>svg.id==='packing-svg'))app.controllers.packing.render();
    });
    mounted.forEach(({ node }) => {
      node.querySelectorAll?.('svg[id]').forEach(svg => {
        app.finishTransition(svg);
        const old = savedCamera.get(svg);
        if (svg.id==='recompute-svg') svg.setAttribute('viewBox','0 0 1040 400');
        else if (old) svg.setAttribute('viewBox', old);
        svg.querySelectorAll('.deck-camera-clipped').forEach(n => n.classList.remove('deck-camera-clipped'));
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
    if(scene.kind==='code')return source('#api-caption');
    if(scene.capabilitySteps)return figures[scene.figure].querySelector('#capability-slide-caption');
    if(scene.comparisonSteps)return figures[scene.figure].querySelector('.comparison-step-caption');
    if(scene.id==='quantization')return figures[scene.figure].querySelector('#quant-detail small');
    if (!scene.figure) return null;
    if(scene.id==='memory-costs'||scene.compact==='working-set')return figures[scene.figure].querySelector('#memory-cost-caption');
    if(scene.id==='allocation')return figures[scene.figure].querySelector('#memory-caption');
    return figures[scene.figure].querySelector('.phase-caption');
  }
  function titleFor(scene,current=false) { if(current&&scene.fusionDirection==='forward')return app.state.fusion==='staged'?'MXFP8 Staged Forward':'MXFP8 Mega Forward';return scene.kind==='cover' ? original.title.textContent : current&&scene.buildTitles ? scene.buildTitles[build] : scene.title; }
  function createShell() {
    shell = document.createElement('div');
    shell.id = 'presentation';
    shell.innerHTML = `
      <header class="deck-toolbar" aria-label="Presentation controls">
        <button id="deck-article" title="Return to the article">Article</button>
        <label class="deck-edition"><span class="sr-only">Presentation</span><select id="deck-edition" aria-label="Choose a presentation"><option value="main">Full Presentation</option><option value="lightning">Lightning</option></select></label>
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
          <header class="deck-slide-header"><div id="deck-brand"></div><a class="deck-repository" id="deck-repository" hidden target="_blank" rel="noopener"></a><span id="deck-section"></span></header>
          <h1 id="deck-title" class="deck-title"></h1>
          <div class="deck-step-heading" id="deck-step-heading"></div><div class="deck-setup" id="deck-setup"></div>
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
    $('deck-edition').value=deck.id;
    $('deck-edition').addEventListener('change',event=>{
      const url=new URL(location.href);url.searchParams.set('view','slides');
      if(event.target.value==='main')url.searchParams.delete('deck');else url.searchParams.set('deck',event.target.value);
      url.searchParams.delete('settings');url.searchParams.delete('print');url.hash='';location.assign(url.href);
    });
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


  function stepSequence(scene){
    return scene.phases || (scene.actions?scene.actions.map(action=>Moe.schedule.indexOf(action)).filter(p=>p>=0):null);
  }

  function range(scene) {
    const c = app.controllers[scene.controller];
    const points=stepSequence(scene);
    if(points?.length)return [points[0],points[points.length-1]];
    return scene.range ? [scene.range[0], scene.range[1] ?? c.count - 1] : [0, 0];
  }
  function slideSteps(scene) {
    if(scene.kind==='code')return app.apiSteps();
    if(scene.builds)return scene.builds.map((_,index)=>index);
    const c=app.controllers[scene.controller];
    if(!c)return [0];
    const sparse=stepSequence(scene);if(sparse)return sparse;
    const [first,last]=range(scene);
    return Array.from({length:last-first+1},(_,i)=>first+i);
  }
  function step(direction, slideOnly = false) {
    const scene = scenes[index], c = app.controllers[scene.controller];
    if(scene.builds&&!slideOnly){const steps=slideSteps(scene),target=steps[steps.indexOf(build)+direction];if(target!=null){setBuild(scene,target);update();return;}}
    if (c && !slideOnly) {
      const points=stepSequence(scene);
      if(points){const target=direction>0?points.find(p=>p>c.phase):[...points].reverse().find(p=>p<c.phase);if(target!=null){c.set(target);update();return;}}
      else{const [first,last]=range(scene);if(direction>0&&c.phase<last){c.set(c.phase+1);update();return;}if(direction<0&&c.phase>first){c.set(c.phase-1);update();return;}}
    }
    const next = index + direction;
    if (next >= 0 && next < scenes.length) show(next, direction < 0);
  }
  /** Apply a presentation build through the same controls used by the article. */
  function setBuild(scene, position) {
    const steps=slideSteps(scene);build=steps.includes(position)?position:steps.at(-1);
    if(scene.capabilitySteps){
      const figure=figures[scene.figure],supported=build===1;
      for(const id of ['capability-table','capability-notes'])figure.querySelector('#'+id).classList.toggle('deck-part',!supported);
      original.support.classList.toggle('deck-part',supported);
      app.phaseCaption('capability-slide-caption',[supported?'Training and Inference by Precision':'CUDA Graphs and Activation Packing',source(supported?'#supported-modes-summary':'#motivation-setup').textContent]);
      return;
    }
    source(scene.builds[build]).click();
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
    const bands=['deck-title','deck-step-heading','deck-setup','deck-summary','deck-scope'].map($);
    const extras=figure?[...figure.children].filter(n=>n.classList.contains('deck-part')&&!n.matches('.diagram-scroll,.table-scroll,.standalone-deck-view')):[];
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
      if(scene.comparisonSteps||scene.capabilitySteps)setBuild(scene,savedBuild);
      if(c){c.phase=phase;c.count=count;c.render();}
      else if(scene.figure==='precision-figure')app.renderPrecision();
      else if(scene.figure==='memory-figure')app.renderMemory();
      else if(scene.figure==='recompute-figure')app.renderRetention();
      else if(scene.kind==='code')app.renderCode();
    };
    calibrating=true;canvas.classList.add('deck-calibrating');c?.stop();
    try {
      app.withoutAnimation(()=>{
        pass();
        if(scene.figure==='precision-figure'){
          for(const format of ['bf16','mxfp8','nvfp4'])for(const direction of ['forward','backward'])for(const outlier of [false,true])for(const byte of [0,7,31]){
            app.state.format=format;app.state.direction=direction;app.state.quantByte=byte;$('quant-outlier').checked=outlier;app.renderPrecision();pass();
          }
        }else if(scene.chapter==='results'&&!scene.report){
          window.ResultsApp.sample(scene.controller,pass);
        }else if(scene.controller==='packing'){
          for(const rank of [0,1])for(const budget of [4096,24576,35840]){app.state.packingRank=rank;app.state.packingBudget=budget;pass();}
        }else if(scene.figure==='recompute-figure'){
          for(const precision of ['bf16','mxfp8'])for(const mode of ['save','recompute']){
            app.state.retentionPrecision=precision;app.state.retention=mode;app.renderRetention();pass();
          }
        }else if(scene.controller==='fusion'){
          for(const pipeline of scene.fusionDirection==='backward'?['mega']:['staged','mega'])for(const chunk of [0,1,2]){app.state.fusion=pipeline;app.state.fusionChunk=chunk;c.count=app.fusionCount(pipeline);pass();}
        }else if(scene.controller==='slots'&&scene.slotDisplay==='reuse'){
          for(const policy of ['stage','microbatch']){app.state.slotPolicy=policy;pass();}
        }else if(scene.controller==='kernel'){
          for(const [m,n,tail,k]of [[128,128,'full',0],[256,256,'tail',3]]){setValue('tile-M',m);setValue('tile-N',n);setValue('tile-tail',tail);setValue('tile-K',k);pass();}
        }else if(scene.controller==='tokens'){
          for(const token of ['A','B','C','D']){app.state.token=token;pass();}
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
    ['deck-title','deck-step-heading','deck-setup','deck-summary','deck-scope'].forEach(id=>clearReservation($(id)));
    $('deck-visual').replaceChildren(); $('deck-controls-content').replaceChildren();
    $('deck-summary').textContent = ''; $('deck-setup').textContent = '';
    canvas.dataset.kind = scene.kind || scene.id;
    canvas.classList.toggle('deck-table-slide', !!scene.table);
    $('deck-slide-select').value = index;
    if (scene.figure) {
      const figure = figures[scene.figure];
      const c = app.controllers[scene.controller];
      move(figure, $('deck-visual'));figure.querySelectorAll('svg[id]').forEach(svg=>window.ArticleLayout?.apply(svg,true));
      if(scene.capabilitySteps){
        move(original.support,figure);
        if(!figure.querySelector('#capability-slide-caption')){
          const caption=document.createElement('div');caption.id='capability-slide-caption';caption.className='phase-caption';caption.hidden=true;figure.append(caption);
        }
      }
      if(scene.report)window.ResultsApp.renderPresentation(scene.report);
      const parts = scene.parts || [c?.svg().id];
      parts.filter(Boolean).forEach(id => part(figure, id));
      const controls = [...figure.children].filter(node => {
        if(scene.compact)return false;
        if(scene.fusionDirection==='backward')return false;
        if (scene.report) return false;
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
      if(scene.fusionDirection){
        source('#fusion-direction [data-direction="'+scene.fusionDirection+'"]').click();
      }
      figure.querySelectorAll('svg[id]').forEach(svg => { if (!savedCamera.has(svg)) savedCamera.set(svg, svg.getAttribute('viewBox')); });
      if(scene.compact){
        const id=scene.compact==='tiers'?'consumption-svg':scene.compact==='working-set'?'memory-svg':'packing-svg';
        figure.querySelector('#'+id).dataset.slideDetail='brief';
        if(scene.memoryExample){
          const example=scene.memoryExample;
          $('vmm-enabled').checked=true;
          $('config-device-factor').value=example.deviceFactor;$('config-total-factor').value=example.totalFactor;
          app.state.memorySlotPolicy='factor';app.state.memorySlotFactor=example.slotFactor;
          source('#memory-slot-policy').querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.policy==='factor')));
        }
        if(scene.compact!=='packing')app.renderMemory();
      }
      if(scene.slotDisplay)document.querySelector(`#slots-display [data-view="${scene.slotDisplay}"]`).click();
      if(scene.display)document.querySelector(`#packing-display [data-view="${scene.display}"]`).click();
      if (scene.prepare) document.querySelector(`#retention-display [data-view="${scene.prepare}"]`).click();
      if(scene.preset)source(scene.preset).click();
      if(scene.builds)setBuild(scene,lastStep?scene.builds.length-1:0);
      if (c) c.set(lastStep ? range(scene)[1] : range(scene)[0]);
      figure.querySelectorAll('svg[id]').forEach(app.finishTransition);
      if (scene.id === 'moe') {
        move(original.mlp, $('deck-visual'));
        original.mlp.classList.add('deck-part');app.renderExpertMath();
      }

    } else if (scene.kind === 'artwork-cover') {
      const image=document.createElement('img');image.className='deck-title-artwork';image.src=scene.asset;image.alt=scene.alt||scene.title;image.width=1920;image.height=1080;
      $('deck-visual').append(image);
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
      const links=document.createElement('nav');links.className='deck-resource-links';links.setAttribute('aria-label','Project resources');
      const repository=document.createElement('a');repository.href=original.repository.href;repository.textContent='Dist-MoE Repository ↗';links.append(repository);
      original.resources.querySelectorAll('a').forEach(sourceLink=>{
        const link=document.createElement('a');link.href=sourceLink.href;link.textContent=sourceLink.querySelector('b').textContent;links.append(link);
      });
      $('deck-visual').append(links);
    } else if (scene.kind === 'code') {
      move(original.code, $('deck-controls-content'));
      const box = document.createElement('div'); box.className = 'deck-code';
      box.innerHTML = '<div class="deck-code-label" id="deck-code-label">Quickstart excerpt</div><pre><code id="deck-code-excerpt"></code></pre><div class="deck-api-plan" id="deck-api-plan" hidden></div>';
      $('deck-visual').append(box);
      if(scene.builds)setBuild(scene,lastStep?scene.builds.length-1:0);
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
    }
    $('deck-fit').hidden = !scene.camera;
    $('deck-fit').textContent = fullView ? 'Focused view · V' : 'Full diagram · V';
    $('deck-fit').setAttribute('aria-pressed',String(fullView));
  }
  function updateCode(scene) {
    const excerpt=app.apiExcerpt(build),blocks=[];let block=[];
    for(const line of excerpt){if(!line.trim()&&block.length){blocks.push(block);block=[];}else block.push(line);}if(block.length)blocks.push(block);
    const columns=build===1?[excerpt]:blocks.length>1?[blocks.slice(0,-1).flatMap((b,i)=>[...(i?['']:[]),...b]),blocks.at(-1)]:[excerpt];
    $('deck-code-excerpt').classList.toggle('deck-code-columns',columns.length>1);
    $('deck-code-excerpt').innerHTML=columns.map(lines=>`<span class="deck-code-column">${lines.map((line,i)=>`<span class="deck-code-line" data-source-line="${i+1}">${app.highlight(line)||' '}</span>`).join('')}</span>`).join('');
    $('deck-code-excerpt').style.fontSize=Math.max(...columns.map(c=>c.length))>16?'22px':'26px';
    let ratio=1;for(const line of $('deck-code-excerpt').querySelectorAll('.deck-code-line')){const range=document.createRange();range.selectNodeContents(line);const width=line.getBoundingClientRect().width;if(width)ratio=Math.max(ratio,range.getBoundingClientRect().width/width);}
    if(ratio>1)$('deck-code-excerpt').style.fontSize=Math.floor(parseFloat($('deck-code-excerpt').style.fontSize)/ratio)+'px';
    $('deck-visual').dataset.codeLines=excerpt.length;
    $('deck-code-label').textContent=$('code-tabs').querySelector('[aria-pressed="true"]').textContent+' · '+['configuration','inspect','execution'][build];
    $('deck-api-plan').hidden=build!==1;$('deck-api-plan').replaceChildren();
    if(build===1){const output=cleanClone(source('#api-plan-output')),toolbar=output.querySelector('.api-console-toolbar');toolbar.replaceChildren(toolbar.querySelector('span'));$('deck-api-plan').append(output);}
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
    if(scene.compact==='tiers')return 'Logical layout · VMM fixes pinned-host backing at setup. Saved activations and separate symmetric communication buffers stay in HBM.';
    if(scene.compact==='working-set')return 'BF16 example · bar widths follow the shared byte model. Scratch is sized per call, not multiplied by the layer count.';
    if(scene.compact==='packing')return 'BF16 · one inspected GPU · both EP ranks agree on each save/replay decision · no host scratch in this example.';
    if(scene.report)return window.ResultsApp.presentationScope(scene.report);
    if(scene.id==='results-metrics')return 'Extra padding/replay FLOPs are not separately instrumented. All routed-operation costs remain in measured iteration time; effective HBM bandwidth is a model.';
    if(scene.chapter==='results')return window.ResultsApp.scope(scene.controller,app.controllers[scene.controller].phase);
    if(scene.chapter==='mok-comparison')return window.ComparisonApp.scope(scene.controller || scene.id.replace('comparison-','compare-'));
    if(scene.id==='capabilities')return build===1?'Supported modes in the public API · NVFP4 uses prepared expert weights.':source('#capability-scope').textContent.split('. ')[0]+'.';
    if(scene.id==='routing')return 'Two ranks, four tokens; output components are illustrative before BF16 rounding.';
    const text=f.querySelector('figcaption').textContent;
    let sentences;
    try{sentences=[...new Intl.Segmenter('en',{granularity:'sentence'}).segment(text)].map(s=>s.segment.trim());}catch{sentences=[text];}
    if(scene.controller==='slots')return sentences.filter(s=>/continuation/.test(s)).join(' ');
    if(scene.id==='lifetimes')return source('#dependency-math-note').textContent;
    if(scene.figure==='precision-figure'){
      const chosen=sentences.filter(s=>app.state.format==='nvfp4'?/NVFP4/.test(s):/finite MXFP8/.test(s)).join(' ');
      const weight=[...f.querySelectorAll('#intermediate-table tbody tr')].find(tr=>tr.firstElementChild.textContent==='W13 / W2 weights');
      return chosen+(app.state.format==='nvfp4'&&weight?' '+weight.lastElementChild.textContent+'.':'');
    }
    if(scene.controller==='kernel')return sentences.filter(s=>/Matrix geometry|Gray rows/.test(s)).join(' ');
    if(scene.figure==='memory-figure')return sentences.find(s=>/These logical bytes/.test(s))||'';
    if(scene.controller==='fusion')return scene.fusionDirection==='backward'?'W2 then W13: separate kernel lifetimes. Each kernel interleaves per-expert DGRAD/WGRAD ranges on shared roles; records are representative, not an instruction trace.':app.state.fusion==='staged'?'Staged forward uses separate W13 and SwiGLU/W2 launches, each with its own CTA lifetimes. Producers and consumers pipeline work within each launch.':'W13/W2 work records share one TMA/MMA warp group and epilogue roles. Record order and exact occupancy depend on configuration.';
    return sentences.find(s=>/illustrative|not measured/.test(s))||'';
  }

  function update() {
    if (!active) return;
    const scene = scenes[index], figure = figures[scene.figure];
    if(scene.kind==='code'&&scene.builds)build=app.state.apiStep;
    $('deck-title').textContent = titleFor(scene,true);
    if(scene.fusionDirection==='forward')$('deck-slide-select').options[index].textContent=String(index+1).padStart(2,'0')+' · '+titleFor(scene,true);
    $('deck-repository').hidden=scene.kind!=='code';
    $('deck-repository').href=original.repository.href;
    $('deck-repository').textContent=original.repository.href;
    $('deck-section').textContent = scene.capabilitySteps ? (build===1?'DIST-MOE MODES':'OSS CAPABILITIES') : figure?.querySelector('.fig-number')?.textContent || source('#' + scene.chapter)?.dataset.title || '';
    $('deck-page').textContent = `${index + 1} / ${scenes.length}`;
    $('deck-reference').href = '#' + scene.chapter;
    const c = app.controllers[scene.controller];
    const steps=slideSteps(scene),at=scene.builds?steps.indexOf(build):c?steps.indexOf(c.phase):0;
    const position=at>=0?at:Math.max(0,steps.findIndex(p=>p>=c.phase));
    $('deck-step').textContent=`Step ${position+1} / ${steps.length}${scene.controller==='slots'?' · '+Moe.schedule[c.phase]:''}`;
    $('deck-back').disabled = index === 0;
    $('deck-next').disabled = index === scenes.length - 1;
    $('deck-controls').hidden = !$('deck-controls-content').children.length;
    $('deck-fit').hidden = !scene.figure || !scene.camera;
    shell.querySelector('.deck-shortcuts').textContent='Arrow keys step through figures · Shift + arrow changes slide'+(scene.camera?' · V toggles focused/full view':'')+' · N opens notes';
    if(scene.report)$('deck-setup').textContent=window.ResultsApp.presentationSetup(scene.report);
    else if (scene.setup) {
      const node = figure?.querySelector(scene.setup) || source(scene.setup);
      $('deck-setup').textContent = node?.textContent || '';
    }
    const caption=captionNode(scene),summary=source(`[data-slide-summary="${scene.id}"]`);
    const dynamic=caption&&scene.id!=='lifetimes'&&scene.id!=='precision';
    const topic=scene.controller==='packing'?caption?.querySelector(scene.id==='layer-packing'||scene.compact==='packing'?'.packing-step-summary':'.decision-step-summary'):null;
    let takeaway=topic?cleanClone(topic):dynamic?firstSentenceFragment(caption,scene.controller==='slots'||scene.chapter==='mok-comparison'?3:2):summary?cleanClone(summary):null;
    if(scene.id==='layer-packing'&&topic){takeaway=document.createDocumentFragment();takeaway.append(firstSentenceFragment(caption.querySelector('.decision-step-summary'),1),document.createTextNode(' '),firstSentenceFragment(topic,1));}
    if(scene.report){takeaway=document.createDocumentFragment();takeaway.append(document.createTextNode(window.ResultsApp.presentationSummary(scene.report)));}
    if(scene.summary){takeaway=document.createDocumentFragment();takeaway.append(document.createTextNode(scene.summary));}
    if(scene.kind==='cover'||scene.kind==='artwork-cover'||scene.kind==='resources'||scene.report)takeaway=null;
    $('deck-step-heading').textContent=scene.subtitle|| (scene.report ? window.ResultsApp.presentationSubtitle(scene.report) : steps.length>1 ? caption?.querySelector('strong')?.textContent.replace(/\.$/,'')||'' : '');
    $('deck-summary').replaceChildren(...(takeaway?[...takeaway.childNodes]:[]));
    $('deck-scope').textContent=scopeFor(scene);
    if (scene.kind === 'code') updateCode(scene);
    if(calibrating)return;
    applyCamera(scene);
    $('deck-notes-caption')?.replaceChildren(...(scene.report ? [document.createTextNode(window.ResultsApp.presentationSummary(scene.report))] : caption ? [cleanClone(caption)] : []));
    const evidence=$('deck-notes-evidence');if(evidence){evidence.replaceChildren();const ids=scene.id==='precision'?['intermediate-table']:scene.id==='memory-costs'?['calculation-table']:scene.figure==='recompute-figure'?['offset-table']:scene.figure==='packing-figure'?['packing-table']:[];if(scene.chapter==='results'&&!scene.report)ids.push(scene.id==='results-metrics'?'metrics-worked-example':scene.controller+'-table');ids.forEach(id=>{const table=$(id);if(table)evidence.append(cleanClone(table));});if(scene.report)document.getElementById(scene.parts[0]).querySelectorAll('table').forEach(t=>evidence.append(cleanClone(t)));}
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
      document.body.classList.remove('presentation-mode');app.renderExpertMath();
      exporting = false;
    }
  }
  function enter(push = false) {
    if (active) return;
    savedFocus = document.activeElement;
    active = true;
    document.title=deck.id==='lightning'?'Lightning · Dist-MoE':original.documentTitle;
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
    document.title=original.documentTitle;
    if(document.fullscreenElement)document.exitFullscreen?.().catch(()=>{});else if(document.webkitFullscreenElement)document.webkitExitFullscreen?.();
    if(speakerWindow&&!speakerWindow.closed)speakerWindow.close();speakerWindow=null;
    shell.remove(); shell = null; canvas = null;
    document.body.classList.remove('presentation-mode');app.renderExpertMath();
    article.classList.remove('deck-source');article.querySelectorAll('svg[id]').forEach(svg=>window.ArticleLayout?.apply(svg)); article.inert = false; article.removeAttribute('aria-hidden');
    setURL(false);
    requestAnimationFrame(() => { $('' + chapter)?.scrollIntoView({ behavior: 'instant', block: 'start' }); (savedFocus || $('present-link')).focus(); });
  }
  $('present-link').addEventListener('click', e => { if(deck.id==='main'){e.preventDefault();enter(true);} });
  $('lightning-link').addEventListener('click',e=>{if(deck.id==='lightning'){e.preventDefault();enter(true);} });
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
  window.Presentation = { enter, exit, show, step, resize, printFrames, scenes, mainScenes, deck, get active() { return active; }, get index() { return index; } };
  if (printMode || new URLSearchParams(location.search).get('view') === 'slides') enter();
})();
