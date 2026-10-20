'use strict';
// One measurement snapshot and one table format serve the article and deck.
(() => {
  const A=Explainer,V=Views,D=ComparisonData,M=D.meta,$=id=>document.getElementById(id);
  const byID=new Map(D.cases.map(r=>[r.id,r])),best=D.pairs.map(p=>({dist:byID.get(p.dist),mok:byID.get(p.mok)}));
  const f=(value,d=3)=>Number(value).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  const precision=fmt=>fmt==='bf16'?'BF16':'MXFP8';
  const mode=ep=>ep<32?'Inference':'Training';
  /** Resolve the same measured condition in both implementations. */
  function pair(ep,fmt,category='single_layer',setting=1){
    const p=best.find(p=>p.dist.ep===ep&&p.dist.precision===fmt&&p.dist.category===category&&(category==='packing'?p.dist.profile===setting:category==='width'?p.dist.F===setting:p.dist.imbalance===setting));
    if(!p)throw new Error('Missing matched iteration-time measurement.');return p;
  }
  const speedup=p=>p.mok.iterationMS/p.dist.iterationMS;
  /** Provider colors describe the two implementations in every comparison. */
  function legend(){return '<span class="cmp-provider dist"><i></i><b>Dist-MoE</b></span><span class="cmp-provider mok"><i></i><b>MoK</b></span><span class="cmp-key-note">Columns: iteration ms ↓ · labels: speedup over MoK · table values: Dist / MoK</span>';}
  const settingName=(category,setting)=>category==='packing'?`Profile ${setting}`:category==='width'?`F = ${V.num(setting)}`:`${setting}× imbalance`;
  const metricSpecs=[
    {key:'usefulTFPerGPU',label:'Useful TFLOP/s / GPU',digits:0,divisor:1,method:'counted work / iteration time'},
    {key:'modeledHBMGBps',label:'HBM model · TB/s',digits:2,divisor:1000,method:'modeled bytes / iteration time'},
    {key:'meanTXGBps',label:'NVLink TX · GB/s',digits:1,divisor:1,method:'measured, mean across GPUs'},
    {key:'meanRXGBps',label:'NVLink RX · GB/s',digits:1,divisor:1,method:'measured, mean across GPUs'}
  ];
  /** A zero-based column scale is shared by both precisions within an EP. */
  function timeScale(ep,category,settings){
    const max=Math.max(...['bf16','mxfp8'].flatMap(fmt=>settings.flatMap(setting=>{const p=pair(ep,fmt,category,setting);return [p.dist.iterationMS,p.mok.iterationMS];})));
    const raw=max/3,magnitude=10**Math.floor(Math.log10(raw)),unit=raw/magnitude,step=(unit<=1?1:unit<=2?2:unit<=2.5?2.5:unit<=5?5:10)*magnitude;
    return {max:Math.ceil(max/step)*step,step};
  }
  /** Keep counted work, modeled HBM traffic, and hardware TX/RX separate. */
  function metricTable(ep,fmt,category,settings){
    const specs=category==='packing'?[...metricSpecs,{key:'peakAllocatedGiB',label:'Peak allocated · GiB',digits:2,divisor:1,method:'allocator, total GPU allocation'}]:metricSpecs;
    const rows=specs.map(metric=>`<tr data-metric="${metric.key}"><th scope="row" title="${metric.method}">${metric.label}</th>${settings.map(setting=>{const p=pair(ep,fmt,category,setting);return `<td data-case="${V.esc(p.dist.id)}"><span class="cmp-metric-pair"><b class="dist" data-provider="dist" data-value="${p.dist[metric.key]}">${f(p.dist[metric.key]/metric.divisor,metric.digits)}</b><span class="cmp-pair-divider" aria-hidden="true">/</span><b class="mok" data-provider="mok" data-value="${p.mok[metric.key]}">${f(p.mok[metric.key]/metric.divisor,metric.digits)}</b></span></td>`;}).join('')}</tr>`).join('');
    return `<table class="cmp-metric-table" aria-label="${precision(fmt)}, EP${ep}; paired Dist-MoE and MoK metrics"><colgroup><col class="cmp-metric-name">${settings.map(()=>'<col>').join('')}</colgroup><thead><tr><th scope="col">Dist / MoK</th>${settings.map(setting=>`<th scope="col">${settingName(category,setting)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
  }
  /** Show absolute iteration times and annotate each paired workload's speedup. */
  function plot(ep,fmt,category,settings,scale){
    const ticks=Array.from({length:Math.round(scale.max/scale.step)+1},(_,i)=>i*scale.step);
    const axis=ticks.map(value=>`<span class="cmp-time-tick" style="bottom:${value/scale.max*100}%"><b>${Number(value.toPrecision(8)).toLocaleString('en-US',{maximumFractionDigits:3})}</b></span>`).join('');
    const columns=settings.map(setting=>{const p=pair(ep,fmt,category,setting),ratio=speedup(p),high=Math.max(p.dist.iterationMS,p.mok.iterationMS)/scale.max*100;
      return `<div class="cmp-time-grouped" data-case="${V.esc(p.dist.id)}" data-iteration-ratio="${ratio}"><span class="cmp-column-speedup" style="bottom:calc(${high}% + 28px)"><b>${f(ratio,2)}×</b><small>speedup</small></span><div class="cmp-time-columns">${['dist','mok'].map(provider=>`<div class="cmp-time-column ${provider}" data-provider="${provider}" data-ms="${p[provider].iterationMS}" style="height:${p[provider].iterationMS/scale.max*100}%"><span class="cmp-ms-label">${f(p[provider].iterationMS,2)}</span></div>`).join('')}</div><span class="cmp-condition">${settingName(category,setting)}</span></div>`;}).join('');
    return `<section class="cmp-precision-panel" data-precision="${fmt}"><h4><b class="cmp-precision-title"><span class="cmp-slide-ep">EP${ep} · ${mode(ep)} · </span>${precision(fmt)}</b><span>Iteration time · ms</span></h4><div class="cmp-panel-body"><div class="cmp-time-chart"><div class="cmp-time-plot" data-y-max="${scale.max}"><div class="cmp-time-axis">${axis}</div><div class="cmp-column-cases" style="--case-count:${settings.length}">${columns}</div></div></div>${metricTable(ep,fmt,category,settings)}</div></section>`;
  }
  /** The article contains every measured EP; slides show EP16 inference/EP64 training. */
  function comparisonView(eps,category,settings){return eps.map(ep=>{const scale=timeScale(ep,category,settings);return `<section class="cmp-ep-report" data-ep="${ep}"><h4 class="cmp-ep-heading">EP${ep} · ${mode(ep)}<span>${category==='packing'?'Ten forward + backward layer calls':ep<32?'One forward layer call':'One forward + backward layer call'}</span></h4><div class="cmp-precision-panels">${['bf16','mxfp8'].map(fmt=>plot(ep,fmt,category,settings,scale)).join('')}</div></section>`;}).join('');}
  /** Every table names its EP/mode and uses the same precision/time columns. */
  function timeTables(eps,category,settings,settingLabel){
    return eps.map(ep=>{const rows=['bf16','mxfp8'].flatMap(fmt=>settings.map(setting=>{const p=pair(ep,fmt,category,setting);return `<tr data-case="${V.esc(p.dist.id)}"><th scope="row">${precision(fmt)}</th><td>${category==='packing'?`Profile ${setting}`:category==='width'?V.num(setting):`${setting}×`}</td><td>${f(p.dist.iterationMS)}</td><td>${f(p.mok.iterationMS)}</td><td class="cmp-speed"><b>${f(speedup(p),3)}×</b></td></tr>`;}));
      return `<div class="cmp-time-group" data-ep="${ep}"><table><caption><strong>EP${ep} · ${mode(ep)}</strong><span>Median iteration time · milliseconds</span></caption><thead><tr><th scope="col">Precision</th><th scope="col">${settingLabel}</th><th scope="col">Dist-MoE (ms)</th><th scope="col">MoK (ms)</th><th scope="col">Speedup over MoK</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
    }).join('');
  }
  /** Confirm the saved/partly-saved pattern holds for both EPs and precisions. */
  function policyPattern(profile){
    const baseline=pair(64,'bf16','packing',profile),partial=baseline.mok.capacity.rank0ReplayRows.map(value=>value>0);
    for(const ep of [32,64])for(const fmt of ['bf16','mxfp8']){const p=pair(ep,fmt,'packing',profile);
      if(p.dist.capacity.capacityRecomputeFlags!==0||p.mok.capacity.rank0ReplayRows.some((value,i)=>(value>0)!==partial[i]))throw new Error('Saved-state policy differs across this comparison.');
    }return {partial,limit:baseline.mok.capacity.effectiveFactor};
  }
  /** Illustrate normalized demand; byte allocation remains precision-dependent. */
  function policyPacking(profile){
    const factors=M.profiles[profile],pattern=policyPattern(profile),slotFactor=pair(64,'bf16','packing',profile).dist.capacity.slotFactor,total=factors.reduce((a,b)=>a+b,0),capacity=factors.length*slotFactor;
    if(total>capacity)throw new Error('The illustrated shared pool cannot retain every layer.');
    for(const ep of [32,64])for(const fmt of ['bf16','mxfp8']){const p=pair(ep,fmt,'packing',profile);if(p.dist.capacity.slotFactor!==slotFactor||Math.abs(p.mok.capacity.effectiveFactor-pattern.limit)>1e-12)throw new Error('Capacity factors differ across the illustrated cases.');}
    let cursor=0;const layers=factors.map((demand,layer)=>{const entry={layer,demand,saved:Math.min(demand,pattern.limit),unused:Math.max(0,pattern.limit-demand),overflow:Math.max(0,demand-pattern.limit),left:cursor/capacity*100,width:demand/capacity*100};cursor+=demand;return entry;});
    if(layers.some((l,i)=>(l.overflow>0)!==pattern.partial[i]))throw new Error('Normalized illustration disagrees with the native replay policy.');
    return {...pattern,layers,total,capacity,free:capacity-total,slotFactor,unused:layers.reduce((a,l)=>a+l.unused,0)};
  }
  /** Show inaccessible spare capacity in MoK and contiguous cross-layer packing. */
  function policyCard(profile){
    const plan=policyPacking(profile),{layers,limit,total,capacity,free}=plan,count=plan.partial.filter(Boolean).length,avg=total/layers.length;
    const bars=layers.map(({layer,demand,saved,unused,overflow})=>`<div class="cmp-layer-bar-wrap" data-layer="${layer}" data-imbalance="${demand}" data-save="${overflow?'partial':'saved'}"><div class="cmp-layer-slot" style="height:${limit/8*100}%" data-capacity="${limit}" data-unused="${unused}"></div><div class="cmp-layer-bar" style="height:${demand/8*100}%"><span class="cmp-layer-retained" style="height:${saved/demand*100}%" data-demand="${saved}"></span>${overflow?`<span class="cmp-layer-overflow" style="height:${overflow/demand*100}%;bottom:${saved/demand*100}%" data-demand="${overflow}"></span>`:''}<span class="cmp-layer-demand ${overflow?'hot':''}">${f(demand,1)}×</span></div><b>L${layer}</b></div>`).join('');
    const packed=layers.map(({layer,demand,left,width})=>`<span class="cmp-packed-layer" data-layer="${layer}" data-demand="${demand}" style="left:${left}%;width:${width}%"></span><i class="cmp-packed-leader tier-${layer%2}" style="left:${left+width/2}%"></i><span class="cmp-packed-label tier-${layer%2}" style="left:clamp(9px,${left+width/2}%,calc(100% - 9px))">L${layer}</span>`).join('');
    return `<div class="cmp-policy-card" data-profile="${profile}"><h4>Imbalance profile ${profile}</h4><p class="cmp-policy-meta">Average <b>${f(avg,2)}×</b> · peak <b>${f(Math.max(...layers.map(l=>l.demand)),1)}×</b></p><h5>MoK · ten separate layer slots</h5><div class="cmp-limit-key"><i></i> Per-layer capacity ≈ ${f(limit,2)} units</div><div class="cmp-layer-plot mok-state" role="img" aria-label="Profile ${profile}: lighter MoK layers leave unused capacity, and ${count} hot layers exceed their separate budgets"><div class="cmp-grid-line top"><span>8×</span></div><div class="cmp-grid-line middle"><span>4×</span></div><div class="cmp-grid-line bottom"><span>0×</span></div><div class="cmp-layer-limit" style="bottom:${limit/8*100}%"></div>${bars}</div><div class="cmp-save-key"><span><i class="saved"></i>Saved</span><span><i class="unused"></i>Unused capacity</span><span><i class="partial"></i>Excess → recompute</span></div><p class="cmp-policy-outcome"><b>${count} hot layers partly saved</b> · ${f(plan.unused,1)} units stay unused</p><h5>Dist-MoE · one ${V.num(capacity)}-unit shared slot</h5><div class="cmp-packed-slot dist-state" data-capacity="${capacity}" data-used="${total}" data-free="${free}" role="img" aria-label="Profile ${profile}: all ten layers use ${f(total,1)} of ${capacity} normalized demand units in one shared slot"><span class="cmp-packed-free" style="left:${total/capacity*100}%;width:${free/capacity*100}%"></span>${packed}</div><p class="cmp-packed-total"><b>${f(total,1)} / ${V.num(capacity)} units</b> packed · ${f(free,1)} unused</p><p class="cmp-policy-outcome dist-outcome"><b>All ten layers saved</b> · 0 capacity-induced recompute</p></div>`;
  }
  /** Render the two workload profiles without a precision or slot selector. */
  function renderPolicy(){A.renderHTML('compare-policy-view',['A','B'].map(policyCard).join(''));}
  function packingCaption(){const fp=pair(64,'mxfp8','packing','B');return ['Packing delivers the largest measured advantage',`EP64 profile B: MXFP8 delivers a ${f(speedup(fp),2)}× speedup and ${f(fp.dist.usefulTFPerGPU,0)} useful TFLOP/s per GPU versus ${f(fp.mok.usefulTFPerGPU,0)} for MoK. Dist-MoE’s shared slot retains eligible state for all ten layers.`];}
  /** Show both precisions and both measured training EPs together. */
  function renderPacking(){A.renderHTML('compare-packing-legend',legend());A.renderHTML('compare-packing-view',comparisonView([32,64],'packing',['A','B']));
    A.renderHTML('compare-packing-table',timeTables([32,64],'packing',['A','B'],'Profile'));A.phaseCaption('compare-packing-caption',packingCaption());
  }
  function scalingCaption(){const a=pair(16,'mxfp8','single_layer',8),b=pair(64,'mxfp8','single_layer',8);return ['The advantage persists under higher imbalance',`At 8× imbalance, MXFP8 delivers a ${f(speedup(a),2)}× speedup over MoK for EP16 inference and a ${f(speedup(b),2)}× speedup for EP64 training. Both precisions deliver a shorter iteration time across the measured EP settings.`];}
  /** All four EP groups are visible; the deck emphasizes its two focus EPs. */
  function renderScaling(){A.renderHTML('compare-scaling-legend',legend());A.renderHTML('compare-scaling-view',comparisonView([8,16,32,64],'single_layer',[1,4,8]));
    A.renderHTML('compare-scaling-table',timeTables([8,16,32,64],'single_layer',[1,4,8],'Imbalance'));A.phaseCaption('compare-scaling-caption',scalingCaption());
  }
  function widthCaption(){return ['More expert work per routed token',`Increasing F adds GEMM work without enlarging routed input/output rows. Useful TFLOP/s rises while each iteration takes longer. Dist-MoE finishes sooner at all three measured widths for EP16 inference and EP64 training.`];}
  /** Width results use the same precision panels, EP key, and time tables. */
  function renderWidth(){A.renderHTML('compare-width-legend',legend());A.renderHTML('compare-width-view',comparisonView([16,64],'width',[1536,3072,6144]));
    A.renderHTML('compare-width-table',timeTables([16,64],'width',[1536,3072,6144],'Width F'));A.phaseCaption('compare-width-caption',widthCaption());
  }
  /** Two presentation steps keep inference and training readable at full size. */
  function presentationSteps(name,category,setting){
    const figure=$(name+'-figure'),steps=document.createElement('div'),caption=document.createElement('div');
    steps.id=name+'-steps';steps.hidden=true;caption.id=name+'-step-caption';caption.className='phase-caption comparison-step-caption';caption.hidden=true;
    steps.innerHTML=[16,64].map(ep=>`<button data-ep="${ep}" aria-pressed="${ep===16}">${mode(ep)} · EP${ep}</button>`).join('');
    figure.append(steps,caption);
    steps.addEventListener('click',event=>{
      const button=event.target.closest('[data-ep]');if(!button)return;const ep=Number(button.dataset.ep);figure.dataset.deckEp=ep;
      steps.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
      const bf=pair(ep,'bf16',category,setting),fp=pair(ep,'mxfp8',category,setting);
      A.phaseCaption(caption.id,[`${mode(ep)} · EP${ep}`,`${category==='width'?`At F = ${V.num(setting)}`:`At ${setting}× imbalance`}, BF16 and MXFP8 deliver ${f(speedup(bf),2)}× and ${f(speedup(fp),2)}× speedup over MoK. Both precisions compare the same routed workload.`]);
    });
    steps.querySelector('button').click();
  }
  presentationSteps('compare-scaling','single_layer',8);
  presentationSteps('compare-width','width',6144);
  renderPolicy();
  renderPacking();renderScaling();renderWidth();
  function scope(name){return name==='compare-policy'?'Widths illustrate normalized layer demand. Byte budgets depend on precision and padding; recorded Dist-MoE capacity flags are zero.':'Matched shapes; one BF16 shared expert. TFLOP/s counts useful work; HBM is modeled; NVLink TX/RX is measured separately.';}
  $('comparison-intro').textContent=`Dist-MoE has a shorter median iteration time in all ${M.matchedWorkloads} matched workloads. Its strongest measured advantage comes from ten-layer training with uneven layer imbalance, where sharing saved-state capacity avoids recompute.`;
  window.ComparisonApp={data:D,pair,speedup,timeScale,metricSpecs,policyPattern,policyPacking,renderPolicy,scope};
})();
