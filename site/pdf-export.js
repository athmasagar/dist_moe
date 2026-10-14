'use strict';
// PDF pages are static captures of the shared presentation, never separate slides.
(() => {
  const app = window.Explainer;
  if (!app || !window.Presentation) return;
  const params = new URLSearchParams(location.search);
  const controls = {
    memorySlotPolicy: 'memory-slot-policy', token: 'token-select',
    memoryPrecision: 'memory-precision', calculation: 'calculation-tabs',
    retentionPrecision: 'recompute-precision', retention: 'recompute-select',
    retentionDisplay: 'retention-display', format: 'format-select',
    direction: 'precision-direction', fusion: 'fusion-select', slotPolicy: 'slot-policy',
    slotsView: 'slots-display', packingDisplay: 'packing-display', code: 'code-tabs'
  };

  /** Carry the current public figure controls into an isolated print tab. */
  function settings() {
    const inputs = Object.fromEntries([...document.querySelectorAll('input[id],select[id]')]
      .filter(node => !node.id.startsWith('deck-') && node.id !== 'slot-budget-slider')
      .map(node => [node.id, { value: node.value, checked: node.checked }]));
    return { state: { ...app.state }, inputs, light: document.documentElement.classList.contains('light') };
  }

  /** Open the only export mode: one page for each selected presentation step. */
  function open() {
    const url = new URL(location.href);
    url.searchParams.set('view', 'print');
    url.searchParams.set('settings', JSON.stringify(settings()));
    url.searchParams.set('print', '1');
    url.hash = '';
    const tab = window.open(url.href, '_blank');
    if (tab) tab.opener = null;
    return !!tab;
  }

  /** Restore only known controls and valid, finite state from the export URL. */
  function applySettings(saved) {
    if (!saved || typeof saved !== 'object') throw new Error('Invalid export settings.');
    for (const [key, value] of Object.entries(saved.state || {})) {
      if (!Object.hasOwn(app.state, key) || typeof value !== typeof app.state[key]) continue;
      if (typeof value === 'number' && (!Number.isFinite(value) || value < 0)) throw new Error('Invalid figure value.');
      if (controls[key]) {
        const buttons = [...document.getElementById(controls[key]).querySelectorAll('button')];
        if (!buttons.some(button => Object.values(button.dataset).includes(value))) throw new Error('Invalid figure mode.');
        buttons.forEach(button => button.setAttribute('aria-pressed', String(Object.values(button.dataset).includes(value))));
      }
      app.state[key] = value;
    }
    if (!Number.isInteger(app.state.quantByte) || app.state.quantByte > 31) throw new Error('Invalid quantization byte.');
    for (const [id, savedInput] of Object.entries(saved.inputs || {})) {
      const node = document.getElementById(id);
      if (!node?.matches('input,select') || id.startsWith('deck-') || id === 'slot-budget-slider') continue;
      if (node.tagName === 'SELECT' && ![...node.options].some(option => option.value === savedInput.value)) throw new Error('Invalid figure selection.');
      if (node.type === 'number' && !Number.isFinite(Number(savedInput.value))) throw new Error('Invalid figure number.');
      node.value = savedInput.value;
      if (node.type === 'checkbox') node.checked = !!savedInput.checked;
    }
    document.documentElement.classList.toggle('light', !!saved.light);
    app.controllers.fusion.count = app.fusionCount();
    app.withoutAnimation(() => {
      for (const controller of Object.values(app.controllers)) { controller.stop(); controller.phase = 0; controller.render(); }
      if (!app.renderMemory()) throw new Error('Check the memory-plan inputs before exporting.');
      app.renderRetention(); app.renderPrecision(); app.renderCode();
    });
  }

  /** Freeze a labeled packet on a visible part of its active edge. */
  function freeze(canvas) {
    const clone = canvas.cloneNode(true);
    const originalSVGs = [...canvas.querySelectorAll('svg[id]')];
    const clonedSVGs = [...clone.querySelectorAll('svg[id]')];
    originalSVGs.forEach((svg, index) => {
      const bounds = svg.viewBox.baseVal;
      const labels = [...svg.querySelectorAll('.edge-payload-label:not(.deck-camera-clipped) .edge-label-bg')].map(node => node.getBBox());
      const dots = [...clonedSVGs[index].querySelectorAll('.packet')];
      [...svg.querySelectorAll('.packet')].forEach((dot, position) => {
        const path = svg.querySelector('#' + CSS.escape(dot.dataset.route));
        if (!path) return;
        for (const fraction of [.5, .35, .65, .2, .8, .1, .9]) {
          const point = path.getPointAtLength(path.getTotalLength() * fraction), margin = 7;
          const inside = point.x >= bounds.x + margin && point.x <= bounds.x + bounds.width - margin && point.y >= bounds.y + margin && point.y <= bounds.y + bounds.height - margin;
          const covered = labels.some(box => point.x >= box.x - margin && point.x <= box.x + box.width + margin && point.y >= box.y - margin && point.y <= box.y + box.height + margin);
          if (inside && !covered) { dots[position].setAttribute('transform', `translate(${point.x},${point.y})`); dots[position].setAttribute('visibility', 'visible'); break; }
        }
      });
    });
    clone.querySelectorAll('.transition-overlay').forEach(node => node.remove());
    clone.querySelectorAll('figure').forEach(figure => [...figure.children].filter(node => !node.classList.contains('deck-part')).forEach(node => node.remove()));
    clone.classList.remove('deck-calibrating');
    clone.style.opacity = '1';
    return clone;
  }

  /** Give every page its own SVG definitions and working article links. */
  function preparePage(frame, page, total) {
    const node = frame.node, ids = new Map();
    [node, ...node.querySelectorAll('[id]')].forEach(element => {
      if (!element.id) return;
      const id = element.id, next = `pdf-${page}-${id}`;
      ids.set(id, next); element.dataset.sourceId = id; element.id = next;
    });
    node.querySelectorAll('*').forEach(element => {
      element.removeAttribute('tabindex'); element.removeAttribute('aria-live');
      for (const attr of [...element.attributes]) {
        if (attr.value.includes('url(#')) element.setAttribute(attr.name, attr.value.replace(/url\(#([^)]+)\)/g, (_, id) => `url(#${ids.get(id) || id})`));
        if (['data-route','aria-labelledby','aria-describedby'].includes(attr.name)) element.setAttribute(attr.name, attr.value.split(' ').map(id => ids.get(id) || id).join(' '));
        if (['href','xlink:href'].includes(attr.name) && attr.value.startsWith('#') && ids.has(attr.value.slice(1))) element.setAttribute(attr.name, '#' + ids.get(attr.value.slice(1)));
      }
    });
    const articleURL = new URL(location.href);
    articleURL.searchParams.delete('view'); articleURL.searchParams.delete('settings'); articleURL.searchParams.delete('print');
    node.querySelectorAll('a[href]').forEach(link => { if (link.getAttribute('href').startsWith('#')) { articleURL.hash = link.getAttribute('href'); link.href = articleURL.href; } });
    node.querySelector('.deck-page').textContent = `Slide ${frame.slide + 1} / ${Presentation.scenes.length} · Step ${frame.step + 1} / ${frame.steps} · ${page} / ${total}`;
    const sheet = document.createElement('section');
    sheet.className = 'deck-print-sheet';
    Object.assign(sheet.dataset, { scene: frame.scene, slide: frame.slide + 1, step: frame.step + 1, steps: frame.steps, phase: frame.phase });
    sheet.setAttribute('aria-label', `Slide ${frame.slide + 1}, step ${frame.step + 1} of ${frame.steps}`);
    sheet.append(node);
    return sheet;
  }

  /** Preserve the shared theme even when article print styles use white paper. */
  function preserveTheme() {
    const root = document.documentElement, style = getComputedStyle(root);
    const values = [...style].filter(name => name.startsWith('--')).map(name => [name, style.getPropertyValue(name)]);
    values.forEach(([name, value]) => root.style.setProperty(name, value));
    root.style.colorScheme = root.classList.contains('light') ? 'light' : 'dark';
  }

  /** Build all static pages, wait for their assets, and offer browser printing. */
  async function render() {
    const status = document.createElement('section');
    status.className = 'deck-print-status'; status.setAttribute('role', 'status');
    status.innerHTML = '<h1>Preparing your PDF</h1><p>One page for every presentation step.</p>';
    document.body.append(status);
    try {
      await document.fonts.ready;
      // Background print tabs can suspend animation frames entirely.
      await new Promise(resolve => setTimeout(resolve, 0));
      const data = params.get('settings');
      if (data?.length > 20000) throw new Error('Export settings are too large.');
      if (data) applySettings(JSON.parse(data));
      preserveTheme();
      const frames = Presentation.printFrames(), pages = document.createElement('main');
      pages.className = 'deck-print'; pages.id = 'deck-pdf-pages';
      pages.setAttribute('aria-label', 'All presentation steps');
      pages.append(...frames.map((frame, index) => preparePage(frame, index + 1, frames.length)));
      const toolbar = document.createElement('header');
      toolbar.className = 'deck-print-toolbar';
      toolbar.innerHTML = `<button type="button">Save as PDF</button><p><strong>${frames.length} pages · all presentation steps · 16:9</strong><br>Choose Save as PDF. Enable background graphics and turn off headers and footers.</p>`;
      toolbar.querySelector('button').addEventListener('click', () => window.print());
      document.body.append(toolbar, pages);
      await Promise.all([...pages.querySelectorAll('img')].map(image => image.decode()));
      document.title = 'Dist-MoE - all presentation steps';
      status.remove(); pages.dataset.ready = 'true';
      if (params.get('print') === '1') setTimeout(() => window.print(), 0);
      return frames.length;
    } catch (error) {
      document.querySelectorAll('.deck-print,.deck-print-toolbar').forEach(node => node.remove());
      status.replaceChildren();
      const heading = document.createElement('h1'); heading.textContent = 'Could not prepare the PDF';
      const detail = document.createElement('p'); detail.textContent = error.message;
      const link = document.createElement('a'); link.textContent = 'Return to the presentation';
      const url = new URL(location.href); url.searchParams.set('view', 'slides'); url.searchParams.delete('settings'); url.searchParams.delete('print'); link.href = url.href;
      status.append(heading, detail, link); status.setAttribute('role', 'alert');
      return false;
    }
  }

  window.DeckPDF = { open, freeze, settings, ready: params.get('view') === 'print' ? render() : Promise.resolve(false) };
})();
