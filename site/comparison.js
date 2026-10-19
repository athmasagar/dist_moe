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
    return `<section class="cmp-precision-panel" data-precision="${fmt}"><h4><b class="cmp-precision-title"><span class="cmp-slide-ep">EP${ep} · ${mode(ep)} · </span>${precision(fmt)}</b><span>Iteration time · ms</span></h4><div class="cmp-time-chart"><div class="cmp-time-plot" data-y-max="${scale.max}"><div class="cmp-time-axis">${axis}</div><div class="cmp-column-cases" style="--case-count:${settings.length}">${columns}</div></div></div>${metricTable(ep,fmt,category,settings)}</section>`;
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
  /** Explain per-layer state ownership without presenting schematic cells as byte sizes. */
  function policyCard(profile){
    const factors=M.profiles[profile],{partial,limit}=policyPattern(profile),count=partial.filter(Boolean).length;
    const avg=factors.reduce((a,b)=>a+b,0)/factors.length;
    const bars=factors.map((factor,i)=>`<div class="cmp-layer-bar-wrap"><div class="cmp-layer-bar ${partial[i]?'hot':''}" style="height:${factor/8*100}%" data-layer="${i}" data-imbalance="${factor}"><span>${f(factor,1)}×</span></div><b>L${i}</b></div>`).join('');
    const cells=partial.map((p,i)=>`<span class="${p?'partial':'saved'}" data-layer="${i}" data-save="${p?'partial':'saved'}" title="Layer ${i}: ${p?'partly saved; remaining state recomputed':'saved'}">L${i}</span>`).join('');
    return `<div class="cmp-policy-card" data-profile="${profile}"><h4>Imbalance profile ${profile}</h4><p class="cmp-policy-meta">Average <b>${f(avg,2)}×</b> · peak <b>${f(Math.max(...factors),1)}×</b></p><div class="cmp-limit-key"><i></i> MoK per-layer limit ≈ ${f(limit,2)}×</div><div class="cmp-layer-plot" role="img" aria-label="Per-layer imbalance factors for profile ${profile}"><div class="cmp-grid-line top"><span>8×</span></div><div class="cmp-grid-line middle"><span>4×</span></div><div class="cmp-grid-line bottom"><span>0×</span></div><div class="cmp-layer-limit" style="bottom:${limit/8*100}%"></div>${bars}</div><div class="cmp-save-key"><span><i class="saved"></i>Saved</span><span><i class="partial"></i>Partly saved</span></div><h5>MoK · a separate budget for each layer</h5><div class="cmp-layer-state mok-state">${cells}</div><p class="cmp-policy-outcome"><b>${count} hot layers partly saved</b> · remaining state recomputed</p><h5>Dist-MoE · one slot shared across layers</h5><div class="cmp-layer-state dist-state">${factors.map((_,i)=>`<span data-layer="${i}" data-save="saved">L${i}</span>`).join('')}</div><p class="cmp-policy-outcome dist-outcome"><b>All ten layers saved</b> · 0 capacity-induced recompute</p></div>`;
  }
  /** Render the two workload profiles without a precision or slot selector. */
  function renderPolicy(){A.renderHTML('compare-policy-view',['A','B'].map(policyCard).join(''));}
  function packingCaption(){const fp=pair(64,'mxfp8','packing','B');return ['Packing delivers the largest measured advantage',`EP64 profile B: MXFP8 delivers a ${f(speedup(fp),2)}× speedup and ${f(fp.dist.usefulTFPerGPU,0)} useful TFLOP/s per GPU versus ${f(fp.mok.usefulTFPerGPU,0)} for MoK. Dist-MoE’s shared slot retains eligible state for all ten layers.`];}
  /** Show both precisions and both measured training EPs together. */
  function renderPacking(){A.renderHTML('compare-packing-legend',legend());A.renderHTML('compare-packing-view',comparisonView([32,64],'packing',['A','B']));
    A.renderHTML('compare-packing-table',timeTables([32,64],'packing',['A','B'],'Profile'));A.phaseCaption('compare-packing-caption',packingCaption());
    const p=pair(64,'mxfp8','packing','B');$('comparison-packing-story').textContent=`At EP64 with profile B, MXFP8 runs the ten-layer iteration in ${f(p.dist.iterationMS)} ms instead of ${f(p.mok.iterationMS)} ms: a ${f(speedup(p),3)}× speedup over MoK and ${f((1-1/speedup(p))*100,1)}% less iteration time. The shared slot avoids capacity-induced recompute in the hot layers.`;
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
  renderPolicy();
  renderPacking();renderScaling();renderWidth();
  for(const [id,pair]of [['comparison-packing',packingCaption()],['comparison-scaling',scalingCaption()],['comparison-width',widthCaption()]])document.querySelector(`[data-slide-summary="${id}"]`).textContent=pair[1];
  const oldCaption=A.getStepCaption;A.getStepCaption=(name,p)=>name==='compare-packing'?packingCaption():name==='compare-scaling'?scalingCaption():name==='compare-width'?widthCaption():oldCaption(name,p);
  /** Static figures need only one layout sample. */
  function sample(name,pass){pass();}
  function scope(name){return name==='compare-policy'?'Saved state is shown schematically. MoK’s partial retention follows its native policy; Dist-MoE’s capacity flags are recorded.':'Matched shapes; one BF16 shared expert. TFLOP/s counts useful work; HBM is modeled; NVLink TX/RX is measured separately.';}
  $('comparison-intro').textContent=`Dist-MoE has a shorter median iteration time in all ${M.matchedWorkloads} matched workloads. Its strongest measured advantage comes from ten-layer training with uneven layer imbalance, where sharing saved-state capacity avoids recompute.`;
  window.ComparisonApp={data:D,pair,speedup,timeScale,metricSpecs,policyPattern,renderPolicy,sample,scope};
})();
