'use strict';
// Calculations, figures, tables, and step copy share the published snapshot.
(() => {
  const A=window.Explainer,V=window.Views,D=StandaloneData,meta=D.meta,$=id=>document.getElementById(id);
  const defaults={resultMode:'inference',resultEP:16,resultInferenceEP:16,resultTrainingEP:32,resultSkewEP:16,resultSkewVariant:'nvfp4:mega',resultWidthEP:16,resultWidthVariant:'nvfp4:mega'};
  Object.assign(A.state,defaults);
  const S=A.state,colors={bf16:'blue',mxfp8:'green',nvfp4:'orange'},skews=[1,4,8],widths=[1536,3072,6144];
  /** Format a displayed number; downloadable data retains original precision. */
  const f=(n,d=3)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  /** Resolve a theme color without changing the measured data. */
  const c=name=>`var(--${name})`;
  /** Draw escaped SVG text in the shared figure coordinate system. */
  const t=(x,y,text,cls='r-label',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${V.esc(String(text))}</text>`;
  /** Draw a line for an axis, range, or explicit dependency. */
  const l=(x1,y1,x2,y2,color='line',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c(color)}" stroke-width="1.5" ${extra}/>`;
  /** Draw a sized region; labels state whether it is data or a projection. */
  const b=(x,y,w,h,color='blue',opacity=.27,extra='')=>`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="3" fill="${c(color)}" fill-opacity="${opacity}" stroke="${c(color)}" ${extra}/>`;
  /** Draw a measured or derived point; the legend distinguishes the series. */
  const dot=(x,y,color,r=5)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c(color)}" class="r-point"/>`;
  /** Resolve the display name of a measured precision and kernel policy. */
  function name(r){return r.precision==='bf16'?'BF16':`${r.precision.toUpperCase()} ${r.pipeline==='mega'?'Mega':'Staged'}`;}
  /** Select one exact experimental control, including its category. */
  function row(ep,variant,category='single_layer',F=3072,imbalance=1){const [precision,pipeline]=variant.split(':');const found=D.cases.find(r=>r.ep===ep&&r.precision===precision&&r.pipeline===pipeline&&r.category===category&&r.F===F&&r.imbalance===imbalance);if(!found)throw new Error('The selected benchmark control is unavailable.');return found;}
  /** List only the formats measured for this workload. */
  function variants(ep){return ['bf16:standard','mxfp8:staged','mxfp8:mega',...(ep<32?['nvfp4:mega']:[])];}
  /** Return balanced default-width cases in a consistent presentation order. */
  function balanced(ep){return variants(ep).map(v=>row(ep,v));}
  /** Count useful GEMM FLOPs; padding and replay are excluded. */
  function work(r,phase='combined'){const multiplier=phase==='backward'?2:phase==='combined'&&r.mode==='training'?3:1;return 6*meta.shape.T*meta.shape.K*meta.shape.D*r.F*multiplier;}
  /** Convert useful work and maximum-rank latency to TFLOP/s. */
  function rate(r,ms=r.layerMS,phase='combined'){return work(r,phase)/ms/1e9;}
  /** Normalize dense useful work to the supplied theoretical datasheet peak. */
  function percent(r,ms=r.layerMS){return 100*rate(r,ms)/meta.theoreticalTFPerGPU[r.precision];}
  /** Preserve observed min/max window endpoints, without inferring confidence. */
  function range(r,phase='combined'){const windows=r.windows[phase];return [Math.min(...windows),Math.max(...windows)];}
  /** Estimate the busiest rank's work rate using conservative elapsed time. */
  function busy(r){return rate(r)*Math.max(...r.receiveRows)/(r.receiveRows.reduce((a,b)=>a+b,0)/r.ep);}
  /** Render a semantic table from the same cases used for the SVG. */
  function table(headers,rows,caption=''){return `<table>${caption?`<caption>${V.esc(caption)}</caption>`:''}<thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(cells=>`<tr>${cells.map((cell,i)=>`<${i?'td':'th'}${i?'':' scope="row"'} data-label="${V.esc(headers[i].replace(/<br\s*\/?\s*>/gi,' ').replace(/<[^>]*>/g,'').replace(/\s+/g,' ').trim())}">${cell}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table>`;}
  /** Show a latency plus the observed five-window range. */
  function msCell(r,phase='combined'){const value=phase==='forward'?r.forwardMS:phase==='backward'?r.backwardMS:r.layerMS,[lo,hi]=range(r,phase);return `${f(value)}<span class="window-range">${f(lo)}–${f(hi)}</span>`;}
  /** Draw a bounded observed range around its measured point. */
  function whisker(x1,x2,y,color){return l(x1,y,x2,y,color,'stroke-width="2"')+l(x1,y-5,x1,y+5,color)+l(x2,y-5,x2,y+5,color);}
  /** Keep step buttons synchronized with the shared controller. */
  function tabs(id,phase){$(id).querySelectorAll('[data-step]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.step)===phase)));}
  /** Draw measured compute or transport against correctly labeled references. */
  function efficiencySVG(rows){
    let svg=t(20,28,'Useful compute / dense theoretical peak','r-head')+t(850,28,'NVLink TX and modeled HBM traffic','r-head');
    const cx=230,cw=405,tx=1030,tw=300,base=342;
    for(let i=0;i<=4;i++){
      const pc=i*25,nv=i*225;
      svg+=l(cx+cw*i/4,69,cx+cw*i/4,base,i===4?'faint':'line')+t(cx+cw*i/4,59,`${pc}%`,'r-small r-muted','middle');
      svg+=l(tx+tw*i/4,69,tx+tw*i/4,base,i===4?'faint':'line')+t(tx+tw*i/4,59,String(nv),'r-small r-muted','middle');
    }
    svg+=t(1570,59,'HBM model','r-small r-muted','end');
    rows.forEach((r,i)=>{
      const yy=100+i*73,color=colors[r.precision],value=percent(r),[lo,hi]=range(r);
      svg+=t(20,yy,name(r),'r-label')+t(20,yy+29,`${f(meta.theoreticalTFPerGPU[r.precision]/1000,1)} PF/s theory`,'r-small r-muted');
      svg+=b(cx,yy-20,cw*value/100,27,color)+whisker(cx+cw*percent(r,hi)/100,cx+cw*percent(r,lo)/100,yy-6,color)+dot(cx+cw*value/100,yy-6,color);
      svg+=t(770,yy,`${f(value,1)}%`,'r-number','end')+t(770,yy+29,`${f(rate(r)/1000,2)} PF/s`,'r-small','end');
      svg+=t(850,yy,name(r),'r-label')+t(850,yy+29,`${f(r.nvlinkTXGBps,0)} → ${f(r.nvlinkMaxTXGBps,0)} GB/s`,'r-small r-muted');
      svg+=b(tx,yy-20,tw*r.nvlinkTXGBps/900,27,color)+whisker(tx+tw*r.nvlinkTXGBps/900,tx+tw*r.nvlinkMaxTXGBps/900,yy-6,color)+dot(tx+tw*r.nvlinkTXGBps/900,yy-6,color);
      svg+=t(1570,yy,`${f(r.analyticalHBMGBps/1000,2)} TB/s`,'r-number','end')+b(1390,yy+18,180,5,'line',.3)+b(1390,yy+18,180*r.analyticalHBMGBps/meta.theoreticalHBMGBps,5,color,.55);
    });
    svg+=t(20,384,'Whiskers: observed timing windows · PF/s = PFLOP/s per GPU','r-small r-muted')+t(850,384,'TX marker: rank mean → max · theory: 900 GB/s TX, 8 TB/s HBM','r-small r-muted');
    return svg;
  }
  /** Generate step-specific interpretation for the hardware comparison. */
  function efficiencyCaption(){const rows=balanced(S.resultEP),bf=rows[0],chosen=rows.find(r=>r.precision==='nvfp4')||rows.find(r=>r.pipeline==='staged');
    const compute=S.resultMode==='training'?`BF16 reaches ${f(percent(bf),1)}% of its dense theoretical peak; ${name(chosen)} delivers ${f(rate(chosen)/1000)} PFLOP/s per GPU and ${f(bf.layerMS/chosen.layerMS,2)}× BF16 useful throughput`:`${name(chosen)} takes ${f(chosen.layerMS)} ms, ${f(bf.layerMS/chosen.layerMS,2)}× faster than BF16, delivering ${f(rate(chosen)/1000)} PFLOP/s per GPU (${f(percent(chosen),1)}% of FP4 dense theory)`;
    return ['Compute and bandwidth in the same measured workload',`${compute}. Its mean NVLink TX is ${f(chosen.nvlinkTXGBps,0)} GB/s (${f(chosen.nvlinkTXGBps/9,1)}% of one-way theory); HBM values are modeled byte rates, with no measured DRAM utilization.`];
  }
  /** Render compute, transport, and the full raw comparison together. */
  function renderEfficiency(){
    const eps=S.resultMode==='inference'?[8,16]:[32,64];if(!eps.includes(S.resultEP))S.resultEP=eps[0];
    $('result-EP').innerHTML=eps.map(ep=>`<option value="${ep}" ${ep===S.resultEP?'selected':''}>${ep}</option>`).join('');$('result-mode').querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===S.resultMode)));
    const rows=balanced(S.resultEP);$('efficiency-config').textContent=`${S.resultMode==='training'?'Forward + backward':'Forward'} · EP${S.resultEP} · balanced · T = 2,048 · D = 7,168 · F = 3,072 · 384 experts · top-6`;
    A.renderSVG('efficiency-svg',efficiencySVG(rows));
    const headers=['Precision / policy',...(S.resultMode==='training'?['Forward<br>ms','Backward<br>ms']:[]),'Layer<br>ms','Useful<br>PFLOP/s / GPU','Dense theory<br>%','Mean TX<br>GB/s','Max TX<br>GB/s','Mean RX<br>GB/s','HBM model<br>TB/s'];
    A.renderHTML('efficiency-table',table(headers,rows.map(r=>[name(r),...(r.mode==='training'?[msCell(r,'forward'),msCell(r,'backward')]:[]),msCell(r),f(rate(r)/1000),f(percent(r),1),f(r.nvlinkTXGBps,1),f(r.nvlinkMaxTXGBps,1),f(r.nvlinkRXGBps,1),f(r.analyticalHBMGBps/1000,3)])));
    A.phaseCaption('efficiency-caption',efficiencyCaption());
  }
  /** Synchronize available variants when the workload changes. */
  function variantSelect(id,key,ep){const available=variants(ep);if(!available.includes(S[key]))S[key]=available.includes('nvfp4:mega')?'nvfp4:mega':'mxfp8:mega';$(id).innerHTML=available.map(v=>`<option value="${v}" ${v===S[key]?'selected':''}>${V.esc(name(row(ep,v)))}</option>`).join('');}
  /** Draw received work and both throughput accounting views together. */
  function skewSVG(rows){
    const mean=meta.shape.T*meta.shape.K,maxRows=mean*8,top=98,base=314,h=base-top;
    let svg='<g class="article-wide-title">'+t(20,28,'All three received-row distributions · same total routes','r-head')+'</g>';
    rows.forEach((r,j)=>{
      const left=110+j*342,right=left+292,barW=(right-left)/r.ep;svg+=`<g class="skew-distribution" data-index="${j}">`;
      svg+=t(left,65,`${r.imbalance}× skew · max ${V.num(Math.max(...r.receiveRows))}`,'r-label');
      for(const value of [0,maxRows/2,maxRows]){const yy=base-value/maxRows*h;svg+=l(left,yy,right,yy);if(j===0)svg+=t(left-10,yy+5,V.num(value),'r-small r-muted','end');}
      r.receiveRows.forEach((count,i)=>{const bh=count/maxRows*h;svg+=b(left+i*barW+1,base-bh,Math.max(1,barW-3),bh,i===0?'orange':'blue',i===0?.7:.4);});
      const my=base-mean/maxRows*h;svg+=l(left,my,right,my,'faint','stroke-dasharray="4 4"');
      [0,r.ep-1].forEach(i=>svg+=t(left+(i+.5)*barW,342,String(i),'r-small r-muted','middle'));svg+='</g>';
    });
    svg+='<g class="skew-throughput">'+t(1120,28,'Useful rate · PFLOP/s','r-head');
    const x0=1190,x1=1550,maxRate=Math.ceil(Math.max(...rows.map(busy))/1000*1.15),y=v=>base-v/maxRate*h,x=v=>x0+(v-1)/7*(x1-x0);
    for(const value of [0,maxRate/2,maxRate])svg+=l(x0,y(value),x1,y(value))+t(x0-12,y(value)+5,f(value,1),'r-small r-muted','end');
    for(const [get,color]of [[r=>rate(r),'blue'],[busy,'orange']]){const points=rows.map(r=>[x(r.imbalance),y(get(r)/1000)]);svg+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')}" stroke="${c(color)}" class="r-guide"/>`;points.forEach(p=>svg+=dot(...p,color,6));}
    skews.forEach(value=>svg+=t(x(value),342,`${value}×`,'r-small r-muted','middle'));
    svg+=dot(1120,381,'blue')+t(1135,389,'Cluster mean','r-small')+dot(1350,381,'orange')+t(1365,389,'Busy rank (derived)','r-small')+'</g><g class="article-wide-title">'+t(20,384,`Rank index · common row scale · dashed line: mean ${V.num(mean)} rows/GPU`,'r-small r-muted')+'</g>';return svg;
  }
  /** Explain the selected skew while keeping measured and derived rates distinct. */
  function skewCaption(){const balancedRow=row(S.resultSkewEP,S.resultSkewVariant),hot=row(S.resultSkewEP,S.resultSkewVariant,'single_layer',3072,8);
    return ['Routing skew and useful throughput',`The busiest GPU’s assigned rows grow ${V.num(Math.max(...balancedRow.receiveRows))} → ${V.num(Math.max(...hot.receiveRows))}; its derived useful rate grows ${f(busy(balancedRow)/1000,2)} → ${f(busy(hot)/1000,2)} PFLOP/s while cluster mean falls as work concentrates. At 8× skew, max-rank TX is ${f(hot.nvlinkMaxTXGBps,0)} GB/s (${f(hot.nvlinkMaxTXGBps/9,1)}% of one-way theory), with zero allocator retries/OOMs in these tested cases.`];
  }
  /** Render every measured distribution with the matching raw numbers. */
  function renderSkew(){variantSelect('skew-variant','resultSkewVariant',S.resultSkewEP);$('skew-EP').value=S.resultSkewEP;const rows=skews.map(i=>row(S.resultSkewEP,S.resultSkewVariant,'single_layer',3072,i));$('skew-config').textContent=`${name(rows[0])} · ${rows[0].mode} · EP${rows[0].ep} · F = 3,072 · total routes fixed · common 8× receive-row ruler`;
    A.renderSVG('skew-svg',skewSVG(rows));A.renderHTML('skew-table',table(['Max / mean<br>rows','Max received<br>rows','Layer<br>ms','Cluster mean<br>PFLOP/s / GPU','Busy rank, derived<br>PFLOP/s','Max-rank TX<br>GB/s','Retries / OOMs'],rows.map(r=>[`${r.imbalance}×`,V.num(Math.max(...r.receiveRows)),msCell(r),f(rate(r)/1000),f(busy(r)/1000),f(r.nvlinkMaxTXGBps,1),`${r.allocationRetries} / ${r.ooms}`])));A.phaseCaption('skew-caption',skewCaption());
  }
  /** Draw proportional feature extents and the three measured width controls. */
  function widthSVG(rows){
    const scale=390/(2*6144);let svg='<g class="width-extents">'+t(20,28,'All expert widths · proportional feature extents','r-head');
    svg+=t(20,65,`x and output keep D = ${V.num(meta.shape.D)}`,'r-small r-muted')+b(470,49,meta.shape.D*scale,21,'blue',.27,'class="width-feature-strip"');
    rows.forEach((r,i)=>{const yy=120+i*81;svg+=t(20,yy,`F = ${V.num(r.F)}`,'r-label')+t(195,yy,`h₁: ${V.num(2*r.F)}`,'r-small')+b(470,yy-17,2*r.F*scale,21,'orange',.27,'class="width-feature-strip"')+t(195,yy+31,`h₂: ${V.num(r.F)}`,'r-small')+b(470,yy+14,r.F*scale,21,'green',.27,'class="width-feature-strip"');});
    svg+=t(20,386,'h₁ width = 2F · h₂ width = F · W₁₃ [2F,D] · W₂ [D,F]','r-small r-muted')+'</g><g class="width-throughput">'+t(880,28,'Useful throughput · PFLOP/s / GPU','r-head');
    const x0=990,x1=1550,base=313,top=93,maxRate=Math.ceil(Math.max(...rows.map(r=>rate(r,range(r)[0])))/1000*1.18),x=F=>x0+(F-1536)/(6144-1536)*(x1-x0),y=pf=>base-pf/maxRate*(base-top),color=colors[rows[0].precision];
    for(const value of [0,maxRate/2,maxRate])svg+=l(x0,y(value),x1,y(value))+t(x0-12,y(value)+5,f(value,1),'r-small r-muted','end');
    const points=rows.map(r=>[x(r.F),y(rate(r)/1000)]);svg+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')}" stroke="${c(color)}" class="r-guide"/>`;
    rows.forEach((r,i)=>{const [lo,hi]=range(r);svg+=l(x(r.F),y(rate(r,hi)/1000),x(r.F),y(rate(r,lo)/1000),color,'stroke-width="2"')+dot(...points[i],color,6)+t(x(r.F),345,V.num(r.F),'r-small r-muted','middle');});
    svg+=t(880,386,`F · ${name(rows[0])} · same logical routed payload`,'r-small r-muted')+'</g>';return svg;
  }
  /** Explain measured work growth, not an interpolated latency prediction. */
  function widthCaption(phase){const rows=widths.map(F=>row(S.resultWidthEP,S.resultWidthVariant,'width',F)),r=rows[phase],bf=row(r.ep,'bf16:standard','width',r.F);
    if(phase===2)return ['More compute per routed byte',`Four times the narrow-width GEMM work takes ${f(r.layerMS/rows[0].layerMS,2)}× its latency and delivers ${f(rate(r)/1000)} PFLOP/s per GPU. At F = ${V.num(r.F)}, ${name(r)} is ${f(bf.layerMS/r.layerMS,2)}× the corresponding BF16 width control.`];
    return [`Measured F = ${V.num(r.F)}`,`The input/output feature width remains D = 7,168 while the expert intermediate width is F = ${V.num(r.F)}. ${name(r)} takes ${f(r.layerMS)} ms and delivers ${f(rate(r)/1000)} PFLOP/s at ${V.num(Math.round(r.forwardFlopsPerLogicalPayloadByteRank0))} FLOPs per logical forward payload byte.`];
  }
  /** Render raw width controls, keeping the repeated F3072 control separate. */
  function renderWidth(phase){variantSelect('width-variant','resultWidthVariant',S.resultWidthEP);$('width-EP').value=S.resultWidthEP;const rows=widths.map(F=>row(S.resultWidthEP,S.resultWidthVariant,'width',F));$('width-config').textContent=`${name(rows[0])} · ${rows[0].mode} · EP${rows[0].ep} · balanced routing · D = 7,168 · same logical activation payload`;
    A.renderSVG('width-svg',widthSVG(rows));A.renderHTML('width-table',table(['Expert width<br>F','Layer<br>ms','Useful<br>PFLOP/s / GPU','Dense theory<br>%','Speedup / BF16<br>same width','Forward FLOPs /<br>logical payload byte','Mean TX<br>GB/s'],rows.map(r=>[V.num(r.F),msCell(r),f(rate(r)/1000),f(percent(r),1),`${f(row(r.ep,'bf16:standard','width',r.F).layerMS/r.layerMS,2)}×`,V.num(Math.round(r.forwardFlopsPerLogicalPayloadByteRank0)),f(r.nvlinkTXGBps,1)])));A.phaseCaption('width-caption',widthCaption(2));
  }
  A.controllers.efficiency=new A.Controller('efficiency',1,renderEfficiency);
  A.controllers.skew=new A.Controller('skew',1,renderSkew);
  A.controllers.width=new A.Controller('width',1,renderWidth);
  $('result-mode').addEventListener('click',event=>{const button=event.target.closest('[data-mode]');if(!button)return;S.resultMode=button.dataset.mode;S.resultEP=S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP'];A.controllers.efficiency.render();});
  $('result-EP').addEventListener('change',event=>{S.resultEP=Number(event.target.value);S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP']=S.resultEP;A.controllers.efficiency.render();});
  [['skew-EP','resultSkewEP','skew',true],['skew-variant','resultSkewVariant','skew',false],['width-EP','resultWidthEP','width',true],['width-variant','resultWidthVariant','width',false]].forEach(([id,key,controller,numeric])=>$(id).addEventListener('change',event=>{S[key]=numeric?Number(event.target.value):event.target.value;A.controllers[controller].render();}));
  $('result-presets').addEventListener('click',event=>{const button=event.target.closest('[data-preset]');if(!button)return;S.resultMode=button.dataset.preset;S.resultEP=S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP'];A.controllers.efficiency.render();});
  /** Reset the new figures together with the existing article controls. */
  function reset(){Object.assign(S,defaults);['efficiency','skew','width'].forEach(name=>A.controllers[name].set(0));}
  const oldReset=A.reset;A.reset=()=>{oldReset();reset();};$('reset').addEventListener('click',reset);
  const oldCaption=A.getStepCaption;A.getStepCaption=(name,phase)=>name==='efficiency'?efficiencyCaption():name==='skew'?skewCaption():name==='width'?widthCaption(2):oldCaption(name,phase);
  /** Sample complete supported modes while presentation reserves a fixed layout. */
  function sample(name,pass){if(name==='efficiency'){for(const mode of ['inference','training'])for(const ep of mode==='inference'?[8,16]:[32,64]){S.resultMode=mode;S.resultEP=ep;pass();}}else if(name==='skew'){for(const ep of [8,16,32,64])for(const variant of variants(ep)){S.resultSkewEP=ep;S.resultSkewVariant=variant;pass();}}else if(name==='width'){for(const ep of [16,64])for(const variant of variants(ep)){S.resultWidthEP=ep;S.resultWidthVariant=variant;pass();}}}
  /** Return a short scope statement matching the current measured/projection view. */
  function scope(name,phase){if(name==='efficiency')return meta.peakNote+' HBM is modeled, not a DRAM counter.';if(name==='skew')return 'Busiest-rank throughput is derived from assigned work and maximum-rank latency. NVLink is measured separately; peaks are theoretical.';if(name==='width')return 'Three measured widths; fixed balanced routing and logical activation payload. HBM weight/intermediate traffic grows with F; peaks are theoretical.';return '';}
  /** Generate the article's numerical prose from the same immutable measurements. */
  function narrative(){
    const bf=row(32,'bf16:standard'),st=row(32,'mxfp8:staged'),balancedNV=row(16,'nvfp4:mega'),hot=row(16,'nvfp4:mega','single_layer',3072,8),small=row(16,'nvfp4:mega','width',1536),large=row(16,'nvfp4:mega','width',6144);
    $('result-compute-story').textContent=`EP32 BF16 training delivers ${f(rate(bf)/1000)} PFLOP/s per GPU (${f(percent(bf),1)}% of dense theory) through the complete forward/backward path. MXFP8 Staged delivers ${f(rate(st)/1000)} PFLOP/s. Its larger arithmetic peak leaves communication and the other layer costs in place, so read latency and useful throughput together.`;
    $('result-skew-story').textContent=`At EP16 with NVFP4, 8× skew gives the busiest rank ${V.num(Math.max(...hot.receiveRows))} rows. Its derived useful rate grows ${f(busy(balancedNV)/1000,2)} → ${f(busy(hot)/1000,2)} PFLOP/s, and maximum TX reaches ${f(hot.nvlinkMaxTXGBps,0)} GB/s. Dist-MoE continues to process the concentrated work; the cluster average falls because the router leaves other GPUs with less work.`;
    $('result-width-story').textContent=`Quadrupling F in the EP16 NVFP4 sweep gives four times the GEMM work for ${f(large.layerMS/small.layerMS,2)}× the latency (${f(small.layerMS)} → ${f(large.layerMS)} ms). Useful throughput reaches ${f(rate(large)/1000)} PFLOP/s per GPU, and speedup over the corresponding BF16 controls rises ${f(row(16,'bf16:standard','width',1536).layerMS/small.layerMS,2)}× → ${f(row(16,'bf16:standard','width',6144).layerMS/large.layerMS,2)}×. The wider experts amortize more compute over the same routed activation payload.`;

  }
  /** Explain effective bandwidth using the cached byte model and timing. */
  function metricsExample(){if(!$('metrics-worked-example')||!meta.metricsExample)return;const e=meta.metricsExample,r=D.cases.find(r=>r.id===e.caseID),bandwidth=e.trainingModelBytes/r.layerMS/1e6;
    $('metrics-worked-example').innerHTML=`<p>EP64, balanced MXFP8 Mega training. The forward model counts ${V.num(e.forwardModelBytes)} logical bytes per GPU. Combined training uses 3× that amount: ${V.num(e.trainingModelBytes)} bytes. Dividing by the ${f(r.layerMS,6)} ms maximum-rank median gives <strong>${f(bandwidth,1)} GB/s effective HBM byte rate</strong>. This is calculated from the model and timer, not measured DRAM traffic.</p>`+table(['Forward byte-model component','Logical bytes'],Object.entries(e.forwardByteBuckets).map(([key,value])=>[({inputs:'Input rows',routed_activations:'Routed activations',routed_weights:'Local routed weights + scales',routing_scores:'Routing scores',shared_work:'Shared-expert work'})[key],V.num(value)]))+`<p>For comparison, the separate rank-0 NVLink TX counter reports ${f(e.rank0TXMiBs,1)} MiB/s = ${f(e.rank0TXMiBs*2**20/1e9,1)} GB/s. The figures report rank means and maxima, so their values can differ from this rank-0 example.</p>`;
  }
  narrative();metricsExample();window.ResultsApp={data:D,row,work,rate,percent,busy,range,sample,scope,reset};
})();
