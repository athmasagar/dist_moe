'use strict';
// One measurement snapshot and one table format serve the article and deck.
(() => {
  const A=Explainer,V=Views,D=ComparisonData,M=D.meta,$=id=>document.getElementById(id);
  const byID=new Map(D.cases.map(r=>[r.id,r])),best=D.pairs.map(p=>({dist:byID.get(p.dist),mok:byID.get(p.mok)}));
  const colors={8:'amber',16:'blue',32:'purple',64:'green'};
  const f=(value,d=3)=>Number(value).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  const precision=fmt=>fmt==='bf16'?'BF16':'MXFP8';
  const mode=ep=>ep<32?'Inference':'Training';
  /** Resolve the same measured condition in both implementations. */
  function pair(ep,fmt,category='single_layer',setting=1){
    const p=best.find(p=>p.dist.ep===ep&&p.dist.precision===fmt&&p.dist.category===category&&(category==='packing'?p.dist.profile===setting:category==='width'?p.dist.F===setting:p.dist.imbalance===setting));
    if(!p)throw new Error('Missing matched iteration-time measurement.');return p;
  }
  const speedup=p=>p.mok.iterationMS/p.dist.iterationMS;
  const text=(x,y,value,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${V.esc(value)}</text>`;
  const line=(x1,y1,x2,y2,color='line',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--${color})" ${extra}/>`;
  /** Keep each EP's color and execution mode identical across figures. */
  function legend(eps){return eps.map(ep=>`<span data-ep="${ep}" data-mode="${mode(ep).toLowerCase()}"><i style="--series-color:var(--${colors[ep]})"></i><b>EP${ep}</b> ${mode(ep).toLowerCase()}</span>`).join('');}
  /** Use the same axes and styles for all speedup comparisons. */
  function plot(fmt,eps,category,settings,xLabel){
    const x0=130,x1=930,top=90,bottom=270;
    const x=i=>category==='packing'?x0+i*(x1-x0)/(settings.length-1):x0+(settings[i]-settings[0])/(settings.at(-1)-settings[0])*(x1-x0);
    const y=value=>bottom-(value-.95)/.55*(bottom-top);
    let s=text(20,38,precision(fmt),'cmp-chart-title')+text(1020,38,'Dist-MoE speedup over MoK','cmp-axis-label','end');
    for(const value of [1,1.2,1.4])s+=line(x0,y(value),x1,y(value),value===1?'muted':'line',value===1?'stroke-dasharray="6 6"':'')+text(x0-20,y(value)+8,`${f(value,1)}×`,'cmp-small muted','end');
    for(const ep of eps){const ps=settings.map(setting=>pair(ep,fmt,category,setting)),c=colors[ep];
      s+=`<g class="cmp-series" data-ep="${ep}"><path d="${ps.map((p,i)=>`${i?'L':'M'}${x(i)},${y(speedup(p))}`).join(' ')}" fill="none" stroke="var(--${c})" stroke-width="3" ${ep>=32?'stroke-dasharray="9 5"':''}/>`;
      ps.forEach((p,i)=>{const xx=x(i),yy=y(speedup(p));
        const shape=ep===16?`<path d="M${xx},${yy-8} l8,8 -8,8 -8,-8Z"/>`:ep===32?`<path d="M${xx},${yy-8} l8,15 -16,0Z"/>`:ep===64?`<rect x="${xx-6}" y="${yy-6}" width="12" height="12"/>`:`<circle cx="${xx}" cy="${yy}" r="6"/>`;
        s+=`<g class="cmp-point" fill="var(--${c})" stroke="var(--paper)" stroke-width="1.2" data-case="${V.esc(p.dist.id)}" data-iteration-ratio="${speedup(p)}" data-x="${xx}" data-y="${yy}">${shape}<title>EP${ep} ${precision(fmt)}: ${f(speedup(p),3)}× · Dist-MoE ${f(p.dist.iterationMS)} ms · MoK ${f(p.mok.iterationMS)} ms</title></g>`;
      });s+='</g>';
    }
    settings.forEach((setting,i)=>s+=text(x(i),310,category==='packing'?`Profile ${setting}`:category==='width'?V.num(setting):`${setting}×`,'cmp-small muted','middle'));
    return s+text(20,342,xLabel+' · MoK iteration time ÷ Dist-MoE iteration time','cmp-footnote muted');
  }
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
  function packingCaption(){const bf=pair(64,'bf16','packing','B'),fp=pair(64,'mxfp8','packing','B');return ['Packing delivers the largest measured advantage',`EP64 profile B: BF16 delivers a ${f(speedup(bf),2)}× speedup and MXFP8 a ${f(speedup(fp),2)}× speedup over MoK. MXFP8 iteration time is ${f(fp.dist.iterationMS)} ms for Dist-MoE versus ${f(fp.mok.iterationMS)} ms for MoK.`];}
  /** Show both precisions and both measured training EPs together. */
  function renderPacking(){A.renderHTML('compare-packing-legend',legend([32,64]));for(const fmt of ['bf16','mxfp8'])A.renderSVG(fmt==='bf16'?'compare-packing-svg':'compare-packing-fp8-svg',plot(fmt,[32,64],'packing',['A','B'],'Imbalance profile'),false);
    A.renderHTML('compare-packing-table',timeTables([32,64],'packing',['A','B'],'Profile'));A.phaseCaption('compare-packing-caption',packingCaption());
    const p=pair(64,'mxfp8','packing','B');$('comparison-packing-story').textContent=`At EP64 with profile B, MXFP8 runs the ten-layer iteration in ${f(p.dist.iterationMS)} ms instead of ${f(p.mok.iterationMS)} ms: a ${f(speedup(p),3)}× speedup over MoK and ${f((1-1/speedup(p))*100,1)}% less iteration time. The shared slot avoids capacity-induced recompute in the hot layers.`;
  }
  function scalingCaption(){const a=pair(16,'mxfp8','single_layer',8),b=pair(64,'mxfp8','single_layer',8);return ['The advantage persists under higher imbalance',`At 8× imbalance, MXFP8 delivers a ${f(speedup(a),2)}× speedup over MoK for EP16 inference and a ${f(speedup(b),2)}× speedup for EP64 training. Both precisions retain a shorter iteration time across the measured EP settings.`];}
  /** All four EP lines are visible; the deck emphasizes its two focus EPs. */
  function renderScaling(){A.renderHTML('compare-scaling-legend',legend([8,16,32,64]));for(const fmt of ['bf16','mxfp8'])A.renderSVG(fmt==='bf16'?'compare-scaling-svg':'compare-scaling-fp8-svg',plot(fmt,[8,16,32,64],'single_layer',[1,4,8],'Imbalance factor'),false);
    A.renderHTML('compare-scaling-table',timeTables([8,16,32,64],'single_layer',[1,4,8],'Imbalance'));A.phaseCaption('compare-scaling-caption',scalingCaption());
  }
  function widthCaption(){const a=pair(16,'mxfp8','width',1536),b=pair(64,'mxfp8','width',6144);return ['The advantage spans all measured expert widths',`Dist-MoE has a shorter iteration time in all twelve width cases. MXFP8 speedup over MoK is ${f(speedup(a),2)}× for EP16 inference at F = 1,536 and ${f(speedup(b),2)}× for EP64 training at F = 6,144.`];}
  /** Width results use the same precision panels, EP key, and time tables. */
  function renderWidth(){A.renderHTML('compare-width-legend',legend([16,64]));for(const fmt of ['bf16','mxfp8'])A.renderSVG(fmt==='bf16'?'compare-width-svg':'compare-width-fp8-svg',plot(fmt,[16,64],'width',[1536,3072,6144],'Expert intermediate width F'),false);
    A.renderHTML('compare-width-table',timeTables([16,64],'width',[1536,3072,6144],'Width F'));A.phaseCaption('compare-width-caption',widthCaption());
  }
  renderPolicy();
  for(const [name,draw]of [['compare-packing',renderPacking],['compare-scaling',renderScaling],['compare-width',renderWidth]])A.controllers[name]=new A.Controller(name,1,draw);
  const oldCaption=A.getStepCaption;A.getStepCaption=(name,p)=>name==='compare-packing'?packingCaption():name==='compare-scaling'?scalingCaption():name==='compare-width'?widthCaption():oldCaption(name,p);
  /** Static figures need only one layout sample. */
  function sample(name,pass){pass();}
  function scope(name){return name==='compare-policy'?'Saved state is shown schematically. MoK’s partial retention follows its native policy; Dist-MoE’s capacity flags are recorded.':'Matched shapes and one BF16 shared expert. Both implementations use CUDA graphs; time is measured per iteration.';}
  $('comparison-intro').textContent=`Dist-MoE has a shorter median iteration time in all ${M.matchedWorkloads} matched workloads. Its strongest measured advantage comes from ten-layer training with uneven layer imbalance, where sharing saved-state capacity avoids recompute.`;
  window.ComparisonApp={data:D,pair,speedup,policyPattern,renderPolicy,sample,scope};
})();
