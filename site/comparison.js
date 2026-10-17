'use strict';
// Charts, interpretations, raw tables, and presentation views share this snapshot.
(() => {
  const A=Explainer,V=Views,D=ComparisonData,$=id=>document.getElementById(id),M=D.meta;
  const defaults={comparePackingEP:64,compareProfile:'B',comparePoolPrecision:'bf16',compareScalingEP:16,compareGraphInferenceEP:16,compareGraphTrainingEP:64,compareEvidenceEP:64,compareEvidenceCategory:'packing',compareEvidenceView:'execution'};
  Object.assign(A.state,defaults);const S=A.state,byID=new Map(D.cases.map(r=>[r.id,r]));
  const f=(value,d=3)=>Number(value).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  const precision=r=>r.precision==='bf16'?'BF16':'MXFP8',pipeline=r=>({standard:'Standard',staged:'Staged',mega:'Mega',native:'Native'})[r.pipeline];
  const label=r=>r.backend==='mok'?'MoK':`Dist ${pipeline(r)}`,color=r=>r.backend==='mok'?'purple':'blue';
  const text=(x,y,value,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${V.esc(value)}</text>`;
  const line=(x1,y1,x2,y2,c='line',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--${c})" ${extra}/>`;
  const rect=(x,y,w,h,c,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="var(--${c})" ${extra}/>`;
  const circle=(x,y,c,extra='')=>`<circle cx="${x}" cy="${y}" r="7" fill="var(--${c})" stroke="var(--paper)" stroke-width="2" ${extra}/>`;
  const table=(headers,rows)=>'<table><thead><tr>'+headers.map(h=>`<th scope="col">${h}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map((cell,i)=>`<${i?'td':'th scope="row"'}>${cell}</${i?'td':'th'}>`).join('')+'</tr>').join('')+'</tbody></table>';
  const best=D.pairs.map(p=>({dist:byID.get(p.dist),mok:byID.get(p.mok),range:p.speedupRange,nonoverlapping:p.nonoverlappingWindows}));
  /** Resolve providers within the same measured workload, including its control category. */
  function pair(ep,fmt,category='single_layer',setting=1){
    const result=best.find(p=>p.dist.ep===ep&&p.dist.precision===fmt&&p.dist.category===category&&(category==='packing'?p.dist.profile===setting:category==='width'?p.dist.F===setting:p.dist.imbalance===setting));
    if(!result)throw new Error('Missing matched comparison.');return result;
  }
  const speedup=p=>p.mok.graphMS/p.dist.graphMS;
  const rowsFor=(ep,category)=>best.filter(p=>p.dist.ep===ep&&p.dist.category===category);
  const settings=(id,key)=>{const node=$(id);node.value=S[key];};
  const buttons=(id,key,value)=>$(id).querySelectorAll(`[data-${key}]`).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset[key]===String(value))));
  const chart=(id,markup)=>A.renderSVG(id,markup,false);

  /** Plot ratios at the measured cases; extrema whiskers are not confidence intervals. */
  function ratios(title,series,xValues,xLabel,dual=false){
    const settings=series[0].pairs.map(p=>p.dist.category==='width'?p.dist.F:p.dist.imbalance);
    const x0=125,x1=920,top=82,bottom=266,x=i=>x0+(settings[i]-settings[0])/(settings.at(-1)-settings[0])*(x1-x0),y=value=>bottom-(value-.9)/.6*(bottom-top);
    let s=text(20,36,title,'cmp-chart-title');
    for(const value of [1,1.2,1.4])s+=line(x0,y(value),x1,y(value),value===1?'muted':'line',value===1?'stroke-dasharray="6 5"':'')+text(x0-18,y(value)+8,`${f(value,1)}×`,'cmp-small muted','end');
    series.forEach(({pairs,color:c,name},j)=>{
      s+=text(dual?615+j*220:1020,36,name,'cmp-small '+c+'-text',dual?'start':'end');
      s+=`<path d="${pairs.map((p,i)=>`${i?'L':'M'}${x(i)},${y(speedup(p))}`).join(' ')}" stroke="var(--${c})" stroke-width="3" fill="none"/>`;
      pairs.forEach((p,i)=>{const yy=y(speedup(p)),[lo,hi]=p.range;
        s+=line(x(i),y(lo),x(i),y(hi),c,'stroke-width="2"')+line(x(i)-6,y(lo),x(i)+6,y(lo),c)+line(x(i)-6,y(hi),x(i)+6,y(hi),c);
        s+=circle(x(i),yy,c,`data-comparison-dist="${V.esc(p.dist.id)}" data-speedup="${speedup(p)}"`)+text(x(i)+(i===0?30:0),yy+(dual&&j===0?36:-18),`${f(speedup(p),2)}×`,'cmp-number '+c+'-text',i===0?'start':'middle');
      });
    });
    xValues.forEach((value,i)=>s+=text(x(i),307,String(value),'cmp-small muted','middle'));
    return s+text(20,333,xLabel+' · MoK / Dist latency · above 1 favors Dist','cmp-footnote muted');
  }

  /** Show routing targets and the native per-layer limit without treating rows as bytes. */
  function loadChart(){
    const targets=M.profiles[S.compareProfile],native=pair(S.comparePackingEP,S.comparePoolPrecision,'packing',S.compareProfile).mok.capacity;
    const x0=95,step=90,base=278,scale=23,cap=native.retainedRowsPerLayer/M.balancedRows;
    let s=text(20,36,'Rank 0 target load · routes / 12,288','cmp-chart-title');
    for(const value of [0,4,8])s+=line(x0-12,base-value*scale,1018,base-value*scale)+text(x0-24,base-value*scale+8,`${value}×`,'cmp-small muted','end');
    targets.forEach((value,i)=>{const x=x0+i*step,c=native.rank0ReplayRows[i]>0?'orange':'purple';s+=rect(x,base-value*scale,58,value*scale,c,'opacity=".55"')+text(x+29,base-value*scale-12,`${f(value,1)}×`,'cmp-number','middle')+text(x+29,307,`L${i}`,'cmp-small muted','middle');});
    s+=line(x0-12,base-cap*scale,1018,base-cap*scale,'purple','stroke-width="2.5" stroke-dasharray="8 5"');
    return s+text(20,333,'Dashed: native retained limit · orange: some native replay','cmp-footnote muted');
  }
  /** Pair actual zero-based latencies; throughput ratios remain separate labels. */
  function packingLatency(){
    const rows=['bf16','mxfp8'].flatMap(fmt=>['A','B'].map(profile=>pair(S.comparePackingEP,fmt,'packing',profile)));
    const max=120,x0=235,w=540;let s=text(20,36,'CUDA-graph latency · milliseconds','cmp-chart-title')+text(745,36,'Dist','cmp-small blue-text')+text(890,36,'MoK','cmp-small purple-text');
    rows.forEach((p,i)=>{const y=83+i*61;s+=text(20,y+9,`${precision(p.dist)} · ${p.dist.profile}`,'cmp-small')+rect(x0,y-17,w*p.dist.graphMS/max,16,'blue')+rect(x0,y+6,w*p.mok.graphMS/max,16,'purple','opacity=".55"')+text(1020,y+10,`${f(speedup(p),2)}×`,'cmp-number blue-text','end');});
    for(const value of [0,40,80,120])s+=text(x0+w*value/max,326,String(value),'cmp-small muted','middle');
    return s;
  }
  function packingCaption(phase){
    const p=pair(S.comparePackingEP,S.comparePoolPrecision,'packing',S.compareProfile),a=p.dist.capacity,n=p.mok.capacity,targets=M.profiles[S.compareProfile],avg=targets.reduce((a,b)=>a+b,0)/10;
    if(phase===0)return ['A per-layer budget cannot borrow from a lighter layer',`Profile ${S.compareProfile} averages ${f(avg,2)}× rank-0 load, but reaches ${f(Math.max(...targets),1)}× in one layer. MoK’s ${V.num(n.retainedRowsPerLayer)}-row native limit requires replay in ${n.rank0ReplayRows.filter(v=>v>0).length} layers on rank 0.`];
    if(phase===1)return ['Ten layers fit in one shared slot',`${precision(p.dist)} retains ${f(a.savedRank0Bytes/2**30,2)} GiB on rank 0 inside a ${f(a.slotBytes/2**30,2)} GiB slot. The recorded capacity-recompute flags are zero across all ${S.comparePackingEP} ranks and ten layers.`];
    const bf=pair(S.comparePackingEP,'bf16','packing',S.compareProfile),fp=pair(S.comparePackingEP,'mxfp8','packing',S.compareProfile);
    return ['The retained-layer workload has the largest measured advantage',`EP${S.comparePackingEP}, profile ${S.compareProfile}: BF16 delivers ${f(speedup(bf),2)}× and MXFP8 ${f(speedup(fp),2)}× MoK’s useful throughput. MXFP8 takes ${f(fp.dist.graphMS)} versus ${f(fp.mok.graphMS)} ms; the avoided-replay contribution was not isolated.`];
  }
  function renderPacking(phase){
    const p=pair(S.comparePackingEP,S.comparePoolPrecision,'packing',S.compareProfile),a=p.dist.capacity,n=p.mok.capacity;
    settings('compare-packing-EP','comparePackingEP');buttons('compare-profile','profile',S.compareProfile);buttons('compare-pool-precision','precision',S.comparePoolPrecision);buttons('compare-packing-steps','step',phase);
    $('compare-packing-view').dataset.phase=phase;
    $('compare-packing-config').textContent=`EP${S.comparePackingEP} training · ten forwards retained → reverse backward · slot factor 2.5 · scratch factor 8 · no host scratch`;
    chart('compare-packing-svg',loadChart());chart('compare-packing-latency-svg',packingLatency());
    A.renderHTML('compare-native-budget',`<p><b>${V.num(n.requestedRowsPerLayer)} requested rows</b> → ${V.num(n.retainedRowsPerLayer)} retained rows/layer, rounded to ${V.num(n.minibatchRows)}-row native minibatches.</p><p class="cmp-muted">Unused capacity stays with its layer. Native replay counts below are source-policy derived.</p>`);
    const used=a.savedRank0Bytes/a.slotBytes*100;
    A.renderHTML('compare-packed-slot',`<h4>Dist-MoE: ten layers share one slot</h4><div class="cmp-byte-bar" role="img" aria-label="${f(used,1)} percent of the measured activation-slot bytes occupied after ten forwards"><span style="width:${used}%">Ten layers retained</span></div><div class="cmp-pool-totals"><span><b>${f(a.savedRank0Bytes/2**30,2)} GiB</b> saved on rank 0</span><span><b>${f(a.slotBytes/2**30,2)} GiB</b> slot capacity</span></div><p class="cmp-zero"><b>0 capacity-recompute flags</b> · all ${S.comparePackingEP} ranks × ten layers</p>`);
    const ps=['bf16','mxfp8'].flatMap(fmt=>['A','B'].map(profile=>pair(S.comparePackingEP,fmt,'packing',profile)));
    A.renderHTML('compare-packing-table',table(['Precision / profile','Dist ms','MoK ms','MoK / Dist'],ps.map(p=>[`${precision(p.dist)} ${pipeline(p.dist)} · ${p.dist.profile}`,f(p.dist.graphMS),f(p.mok.graphMS),`<b>${f(speedup(p),3)}×</b>`])));
    A.renderHTML('compare-capacity-table',table(['Precision / profile','Dist slot GiB','Dist saved rank 0 GiB','Dist capacity flags','MoK replaying layers, rank 0','MoK replay rows, rank 0','Allocated peak GiB: Dist / MoK'],ps.map(p=>[`${precision(p.dist)} · ${p.dist.profile}`,f(p.dist.capacity.slotBytes/2**30,2),f(p.dist.capacity.savedRank0Bytes/2**30,2),'0',p.mok.capacity.rank0ReplayRows.filter(v=>v>0).length,V.num(p.mok.capacity.rank0ReplayRows.reduce((a,b)=>a+b,0)),`${f(p.dist.peakAllocatedGiB,2)} / ${f(p.mok.peakAllocatedGiB,2)}`])));
    $('compare-layer-rows-heading').textContent=`EP${S.comparePackingEP} · profile ${S.compareProfile} · rank 0 · ${precision(p.dist)}`;
    A.renderHTML('compare-layer-rows-table',table(['Layer','Target routes','Native rounded schedule rows','Native row budget','Policy-derived replay rows'],M.profiles[S.compareProfile].map((factor,i)=>[`L${i}`,V.num(Math.round(factor*M.balancedRows)),V.num(n.rank0ScheduleRows[i]),V.num(n.retainedRowsPerLayer),V.num(n.rank0ReplayRows[i])])));
    A.phaseCaption('compare-packing-caption',packingCaption(phase));
    const fp=pair(64,'mxfp8','packing','B'),bf=pair(64,'bf16','packing','B');
    $('comparison-packing-story').textContent=`EP64 profile B makes the effect concrete: MXFP8 takes ${f(fp.dist.graphMS)} ms versus MoK’s ${f(fp.mok.graphMS)} ms—${f(speedup(fp),3)}× useful throughput and ${f((1-1/speedup(fp))*100,1)}% less latency. BF16 delivers ${f(speedup(bf),3)}×. This capacity flexibility does not guarantee lower total allocated HBM: MXFP8 peaks at ${f(fp.dist.peakAllocatedGiB,2)} GiB for Dist versus ${f(fp.mok.peakAllocatedGiB,2)} GiB for MoK. The saved-state policy and the total allocation answer different questions.`;
  }

  function scalingCaption(){const ep=S.compareScalingEP,bf=pair(ep,'bf16'),fp=pair(ep,'mxfp8'),hot=pair(ep,'mxfp8','single_layer',8);return [`EP${ep} ${fp.dist.mode}: balanced and concentrated routing`,`Balanced BF16 is ${f(speedup(bf),2)}× and MXFP8 ${f(speedup(fp),2)}× faster on the overall latency metric. At 8× imbalance, MXFP8 retains a ${f(speedup(hot),2)}× advantage; absolute layer time still grows with the concentrated work.`];}
  function renderScaling(){
    const ep=S.compareScalingEP;settings('compare-scaling-EP','compareScalingEP');const mode=ep<32?'inference · forward':'training · forward + backward';
    $('compare-scaling-config').textContent=`EP${ep} ${mode} · one layer · T = 2,048 · D = 7,168 · F = 3,072 · top-6 + one BF16 shared expert`;
    for(const fmt of ['bf16','mxfp8'])chart(fmt==='bf16'?'compare-scaling-svg':'compare-scaling-fp8-svg',ratios(fmt==='bf16'?'BF16':'MXFP8',[{pairs:[1,4,8].map(i=>pair(ep,fmt,'single_layer',i)),color:fmt==='bf16'?'blue':'green',name:'MoK / Dist'}],['1×','4×','8×'],'Max-rank / mean routed rows'));
    const ps=['bf16','mxfp8'].flatMap(fmt=>[1,4,8].map(i=>pair(ep,fmt,'single_layer',i)));
    A.renderHTML('compare-scaling-table',table(['Precision / imbalance','Dist pipeline','Dist graph ms','MoK graph ms','MoK / Dist','Timing windows'],ps.map(p=>[`${precision(p.dist)} · ${p.dist.imbalance}×`,pipeline(p.dist),f(p.dist.graphMS),f(p.mok.graphMS),`<b>${f(speedup(p),3)}×</b>`,p.nonoverlapping?'Non-overlapping':'Overlap'])));
    A.phaseCaption('compare-scaling-caption',scalingCaption());
  }
  function widthCaption(){const fp=pair(16,'mxfp8','width',1536),tr=pair(64,'mxfp8','width',6144);return ['The advantage spans the measured expert widths',`Dist has a lower median in all twelve width settings. MXFP8 is ${f(speedup(fp),2)}× MoK at EP16, F = 1,536, and ${f(speedup(tr),2)}× at EP64, F = 6,144; the best Staged/Mega choice depends on geometry.`];}
  function renderWidth(){
    for(const ep of [16,64])chart(ep===16?'compare-width-svg':'compare-width-training-svg',ratios(`EP${ep} ${ep===16?'inference':'training'}`,['bf16','mxfp8'].map(fmt=>({pairs:[1536,3072,6144].map(F=>pair(ep,fmt,'width',F)),color:fmt==='bf16'?'blue':'green',name:fmt==='bf16'?'BF16':'MXFP8'})),['1,536','3,072','6,144'],'Expert intermediate width F',true));
    const rows=['bf16','mxfp8'].flatMap(fmt=>[1536,3072,6144].map(F=>{const a=pair(16,fmt,'width',F),b=pair(64,fmt,'width',F);return [precision(a.dist)+' · '+V.num(F),`${f(a.dist.graphMS)} <small>${pipeline(a.dist)}</small>`,f(a.mok.graphMS),`<b>${f(speedup(a),3)}×</b>`,`${f(b.dist.graphMS)} <small>${pipeline(b.dist)}</small>`,f(b.mok.graphMS),`<b>${f(speedup(b),3)}×</b>`];}));
    A.renderHTML('compare-width-table',table(['Precision / F','EP16 Dist ms','EP16 MoK ms','Ratio','EP64 Dist ms','EP64 MoK ms','Ratio'],rows));A.phaseCaption('compare-width-caption',widthCaption());
  }
  function graphsChart(ep){
    const ps=['bf16','mxfp8'].map(fmt=>pair(ep,fmt)),rs=ps.flatMap(p=>[p.dist,p.mok]);
    const max=Math.ceil(Math.max(...rs.map(r=>r.eagerMS))),x0=290,w=480;let s=text(20,36,`EP${ep} ${ep<32?'inference':'training'}`,'cmp-chart-title')+text(610,36,'Filled: graph · outline: eager','cmp-small muted');
    rs.forEach((r,i)=>{const y=88+i*61,c=color(r);s+=text(20,y+8,`${precision(r)} ${r.backend==='mok'?'MoK':'Dist'}`,'cmp-small')+`<rect x="${x0}" y="${y-20}" width="${w*r.eagerMS/max}" height="29" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.5"/>`+rect(x0,y-13,w*r.graphMS/max,15,c,'opacity=".7"')+text(1020,y+8,`${f(r.eagerMS/r.graphMS,2)}×`,'cmp-number '+c+'-text','end');});
    for(const value of [0,max/2,max])s+=text(x0+w*value/max,325,`${f(value,1)} ms`,'cmp-small muted','middle');return s;
  }
  function graphsCaption(){const a=pair(S.compareGraphInferenceEP,'mxfp8').dist,b=pair(S.compareGraphTrainingEP,'mxfp8').dist;return ['The graph execution benefit is much larger for Dist',`With the same pipeline and allocation, MXFP8 improves ${f(a.eagerMS/a.graphMS,2)}× at EP${a.ep} inference and ${f(b.eagerMS/b.graphMS,2)}× at EP${b.ep} training. Native MoK’s balanced graph/eager controls change by about 1.02–1.03×.`];}
  function renderGraphs(){
    settings('compare-graphs-inference-EP','compareGraphInferenceEP');settings('compare-graphs-training-EP','compareGraphTrainingEP');
    $('compare-graphs-config').textContent=`Balanced · EP${S.compareGraphInferenceEP} inference / EP${S.compareGraphTrainingEP} training · fixed graph → eager order · same pipeline and allocation`;
    chart('compare-graphs-svg',graphsChart(S.compareGraphInferenceEP));chart('compare-graphs-training-svg',graphsChart(S.compareGraphTrainingEP));
    const ps=[S.compareGraphInferenceEP,S.compareGraphTrainingEP].flatMap(ep=>['bf16','mxfp8'].map(fmt=>pair(ep,fmt)));
    A.renderHTML('compare-graphs-table',table(['EP / precision','Dist pipeline','Dist eager ms','Dist graph ms','Dist eager / graph','MoK eager ms','MoK graph ms','MoK eager / graph'],ps.map(p=>[`EP${p.dist.ep} · ${precision(p.dist)}`,pipeline(p.dist),f(p.dist.eagerMS),f(p.dist.graphMS),`<b>${f(p.dist.eagerMS/p.dist.graphMS,2)}×</b>`,f(p.mok.eagerMS),f(p.mok.graphMS),`${f(p.mok.eagerMS/p.mok.graphMS,2)}×`])));A.phaseCaption('compare-graphs-caption',graphsCaption());
  }
  /** Keep the detailed numerical evidence optional and filterable, including both pipelines. */
  function renderEvidence(){
    settings('compare-evidence-EP','compareEvidenceEP');settings('compare-evidence-category','compareEvidenceCategory');buttons('compare-evidence-tabs','view',S.compareEvidenceView);
    const rs=D.cases.filter(r=>r.ep===S.compareEvidenceEP&&r.category===S.compareEvidenceCategory).sort((a,b)=>a.precision.localeCompare(b.precision)||(a.profile||'').localeCompare(b.profile||'')||a.imbalance-b.imbalance||a.F-b.F||a.backend.localeCompare(b.backend)||a.pipeline.localeCompare(b.pipeline));
    const name=r=>`${precision(r)} ${label(r)} · ${r.profile||`${r.imbalance}× · F${V.num(r.F)}`}`;
    const markup=S.compareEvidenceView==='execution'?table(['Implementation / setting','Graph overall ms','Eager overall ms','Forward ms','Backward ms','Useful TFLOP/s / GPU'],rs.map(r=>[name(r),f(r.graphMS),f(r.eagerMS),f(r.forwardMS),r.backwardMS==null?'—':f(r.backwardMS),f(r.usefulTFPerGPU,1)])):table(['Implementation / setting','Mean TX GB/s','Mean RX GB/s','HBM byte model GB/s','Allocated peak GiB','Reserved peak GiB'],rs.map(r=>[name(r),f(r.meanTXGBps,1),f(r.meanRXGBps,1),f(r.modeledHBMGBps,1),f(r.peakAllocatedGiB,2),r.peakReservedBytes==null?'—':f(r.peakReservedBytes/2**30,2)]));
    A.renderHTML('compare-evidence-table',rs.length?markup:'<p>This campaign has no width or packing case at this GPU count. Choose EP16/64 for width or EP32/64 for packing.</p>');
  }
  ['compare-packing','compare-scaling','compare-width','compare-graphs'].forEach(name=>{A.controllers[name]=new A.Controller(name,name==='compare-packing'?3:1,name==='compare-packing'?renderPacking:name==='compare-scaling'?renderScaling:name==='compare-width'?renderWidth:renderGraphs);});
  $('compare-packing-steps').addEventListener('click',e=>{const b=e.target.closest('[data-step]');if(b)A.controllers['compare-packing'].set(Number(b.dataset.step));});
  const change=(id,key,render,numeric=true)=>$(id).addEventListener('change',e=>{S[key]=numeric?Number(e.target.value):e.target.value;render();});
  change('compare-packing-EP','comparePackingEP',()=>A.controllers['compare-packing'].render());
  [['compare-profile','profile','compareProfile'],['compare-pool-precision','precision','comparePoolPrecision']].forEach(([id,attr,key])=>$(id).addEventListener('click',e=>{const b=e.target.closest(`[data-${attr}]`);if(b){S[key]=b.dataset[attr];A.controllers['compare-packing'].render();}}));
  change('compare-scaling-EP','compareScalingEP',renderScaling);
  $('compare-scaling-presets').addEventListener('click',e=>{const b=e.target.closest('[data-ep]');if(b){S.compareScalingEP=Number(b.dataset.ep);renderScaling();}});
  $('compare-packing-presets').addEventListener('click',()=>{S.comparePackingEP=64;A.controllers['compare-packing'].render();});
  $('compare-graphs-presets').addEventListener('click',()=>{S.compareGraphInferenceEP=16;S.compareGraphTrainingEP=64;renderGraphs();});
  change('compare-graphs-inference-EP','compareGraphInferenceEP',renderGraphs);change('compare-graphs-training-EP','compareGraphTrainingEP',renderGraphs);
  change('compare-evidence-EP','compareEvidenceEP',renderEvidence);change('compare-evidence-category','compareEvidenceCategory',renderEvidence,false);
  $('compare-evidence-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-view]');if(b){S.compareEvidenceView=b.dataset.view;renderEvidence();}});
  function reset(){Object.assign(S,defaults);['compare-packing','compare-scaling','compare-width','compare-graphs'].forEach(name=>A.controllers[name].set(name==='compare-packing'?2:0));renderEvidence();}
  const oldReset=A.reset;A.reset=()=>{oldReset();reset();};$('reset').addEventListener('click',reset);
  const oldCaption=A.getStepCaption;A.getStepCaption=(name,p)=>name==='compare-packing'?packingCaption(p):name==='compare-scaling'?scalingCaption():name==='compare-width'?widthCaption():name==='compare-graphs'?graphsCaption():oldCaption(name,p);
  /** Reserve layout across the article's supported controls, restoring state afterward. */
  function sample(name,pass){
    if(name==='compare-packing')for(const ep of [32,64])for(const profile of ['A','B'])for(const fmt of ['bf16','mxfp8']){S.comparePackingEP=ep;S.compareProfile=profile;S.comparePoolPrecision=fmt;pass();}
    if(name==='compare-scaling')for(const ep of [8,16,32,64]){S.compareScalingEP=ep;pass();}
    if(name==='compare-graphs')for(const a of [8,16])for(const b of [32,64]){S.compareGraphInferenceEP=a;S.compareGraphTrainingEP=b;pass();}
    if(name==='compare-width')pass();
  }
  function scope(name){return name==='compare-packing'?'Matched CUDA graphs and one BF16 shared expert. MoK replay is source-policy derived; factors are not equal byte budgets.':name==='compare-graphs'?'Same pipeline and allocation within each graph/eager pair. The experiment measures the full graph treatment, not individual overheads.':'Matched CUDA graphs and one BF16 shared expert. Whiskers are observed-window extrema, not confidence intervals; best measured Dist graph pipeline per case.';}
  $('comparison-intro').textContent=`Dist-MoE has a lower overall graph median in all ${M.matchedWorkloads} matched workloads, including all ${M.distPipelinePairs} individual Dist pipeline comparisons. The strongest gains are in retained multi-layer workloads with changing routing imbalance. In 43 workloads the observed timing ranges do not overlap; one overlaps. These ranges are not confidence intervals.`;
  renderEvidence();window.ComparisonApp={data:D,pair,speedup,rowsFor,sample,scope,reset,renderEvidence};
})();
