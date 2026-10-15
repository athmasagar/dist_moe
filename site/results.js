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
  function table(headers,rows,caption=''){return `<table>${caption?`<caption>${V.esc(caption)}</caption>`:''}<thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(cells=>`<tr>${cells.map((cell,i)=>`<${i?'td':'th'}${i?'':' scope="row"'}>${cell}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table>`;}
  /** Show a latency plus the observed five-window range. */
  function msCell(r,phase='combined'){const value=phase==='forward'?r.forwardMS:phase==='backward'?r.backwardMS:r.layerMS,[lo,hi]=range(r,phase);return `${f(value)}<span class="window-range">${f(lo)}–${f(hi)}</span>`;}
  /** Draw a bounded observed range around its measured point. */
  function whisker(x1,x2,y,color){return l(x1,y,x2,y,color,'stroke-width="2"')+l(x1,y-5,x1,y+5,color)+l(x2,y-5,x2,y+5,color);}
  /** Keep step buttons synchronized with the shared controller. */
  function tabs(id,phase){$(id).querySelectorAll('[data-step]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.step)===phase)));}
  /** Draw measured compute or transport against correctly labeled references. */
  function efficiencySVG(rows,phase){
    const transport=phase===1,x0=260,x1=transport?800:892,scale=(x1-x0)/(transport?900:100),bottom=285;
    let svg=t(16,25,transport?'Measured NVLink TX; modeled HBM traffic':'Useful GEMM work / complete layer time','r-head');
    for(let i=0;i<=4;i++){const value=(transport?900:100)*i/4,x=x0+value*scale;svg+=l(x,55,x,bottom,i===4?'faint':'line',i===4?'class="r-theory"':'')+t(x,48,transport?String(value):`${value}%`,'r-small r-muted','middle');}
    if(transport)svg+=t(1018,25,'HBM model','r-head','end')+t(1018,48,'0 → 8 TB/s theory','r-tiny r-muted','end');
    rows.forEach((r,i)=>{const y=84+i*59,col=colors[r.precision];svg+=t(16,y,name(r))+t(16,y+24,transport?'TX: rank mean → max':`Theory: ${f(meta.theoreticalTFPerGPU[r.precision]/1000,1)} PFLOP/s`,'r-small r-muted');
      const value=transport?r.nvlinkTXGBps:percent(r);svg+=b(x0,y-18,value*scale,24,col);
      if(transport){svg+=whisker(x0+r.nvlinkTXGBps*scale,x0+r.nvlinkMaxTXGBps*scale,y-6,col)+dot(x0+r.nvlinkTXGBps*scale,y-6,col)+t(1018,y,`${f(r.analyticalHBMGBps/1000,2)} TB/s`,'r-number','end')+b(876,y+8,142,4,'line',.3)+b(876,y+8,142*r.analyticalHBMGBps/meta.theoreticalHBMGBps,4,col,.55);}
      else{const [lo,hi]=range(r);svg+=whisker(x0+percent(r,hi)*scale,x0+percent(r,lo)*scale,y-6,col)+dot(x0+value*scale,y-6,col)+t(1024,y,`${f(value,1)}%`,'r-number','end');}
    });
    if(rows.length===3)svg+=t(16,278,'NVFP4 is inference-only.','r-small r-muted');
    svg+=t(16,319,transport?'TX axis: GB/s · 900 GB/s one-way theoretical peak · HBM theory: 8 TB/s':'100% = dense theoretical peak for each precision · whiskers: observed timing windows','r-small r-muted');return svg;
  }
  /** Generate step-specific interpretation for the hardware comparison. */
  function efficiencyCaption(phase){const rows=balanced(S.resultEP),bf=rows[0],chosen=rows.find(r=>r.precision==='nvfp4')||rows.find(r=>r.pipeline==='staged');
    if(phase===1)return ['Transport and memory context',`${name(chosen)} measures ${f(chosen.nvlinkTXGBps,0)} GB/s mean TX, ${f(chosen.nvlinkTXGBps/9,1)}% of the one-way theoretical NVLink peak. The HBM column is modeled useful traffic, not a DRAM-utilization counter.`];
    if(S.resultMode==='training')return ['Useful training compute through the distributed path',`BF16 delivers ${f(percent(bf),1)}% of the dense theoretical Tensor Core peak through complete forward/backward execution. ${name(chosen)} delivers ${f(rate(chosen)/1000)} PFLOP/s per GPU and ${f(bf.layerMS/chosen.layerMS,2)}× the BF16 useful throughput.`];
    return ['Useful inference work and the precision-specific ceiling',`${name(chosen)} completes the routed layer in ${f(chosen.layerMS)} ms, ${f(bf.layerMS/chosen.layerMS,2)}× faster than BF16. Its ${f(rate(chosen)/1000)} PFLOP/s per GPU is ${f(percent(chosen),1)}% of the larger dense theoretical FP4 peak.`];
  }
  /** Render hardware data without changing its canonical SVG viewBox. */
  function renderEfficiency(phase){
    const eps=S.resultMode==='inference'?[8,16]:[32,64];if(!eps.includes(S.resultEP))S.resultEP=eps[0];
    $('result-EP').innerHTML=eps.map(ep=>`<option value="${ep}" ${ep===S.resultEP?'selected':''}>${ep}</option>`).join('');$('result-mode').querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===S.resultMode)));
    const rows=balanced(S.resultEP);$('efficiency-config').textContent=`${S.resultMode==='training'?'Forward + backward':'Forward'} · EP${S.resultEP} · balanced · T = 2,048 · D = 7,168 · F = 3,072 · 384 experts · top-6`;
    A.renderSVG('efficiency-svg',efficiencySVG(rows,phase));
    if(phase===0){const headers=['Precision / policy',...(S.resultMode==='training'?['Forward<br>ms','Backward<br>ms']:[]),'Layer<br>ms','Useful<br>PFLOP/s / GPU','Dense theory<br>%'];A.renderHTML('efficiency-table',table(headers,rows.map(r=>[name(r),...(r.mode==='training'?[msCell(r,'forward'),msCell(r,'backward')]:[]),msCell(r),f(rate(r)/1000),f(percent(r),1)])));}
    else A.renderHTML('efficiency-table',table(['Precision / policy','Mean TX<br>GB/s','Max-rank TX<br>GB/s','Mean RX<br>GB/s','Mean TX / theory<br>%','HBM model<br>TB/s'],rows.map(r=>[name(r),f(r.nvlinkTXGBps,1),f(r.nvlinkMaxTXGBps,1),f(r.nvlinkRXGBps,1),f(r.nvlinkTXGBps/9,1),f(r.analyticalHBMGBps/1000,3)])));
    tabs('efficiency-tabs',phase);A.phaseCaption('efficiency-caption',efficiencyCaption(phase));
  }
  /** Synchronize available variants when the workload changes. */
  function variantSelect(id,key,ep){const available=variants(ep);if(!available.includes(S[key]))S[key]=available.includes('nvfp4:mega')?'nvfp4:mega':'mxfp8:mega';$(id).innerHTML=available.map(v=>`<option value="${v}" ${v===S[key]?'selected':''}>${V.esc(name(row(ep,v)))}</option>`).join('');}
  /** Draw received work and both throughput accounting views together. */
  function skewSVG(rows,phase){
    const current=rows[phase],n=current.ep,mean=meta.shape.T*meta.shape.K,base=252,top=70,left=88,right=488,barWidth=(right-left)/n,maxRows=mean*8;
    let svg=t(16,25,'Actual received expert rows','r-head')+t(565,25,'Useful rate per GPU · PFLOP/s','r-head');
    for(const value of [0,maxRows/2,maxRows]){const y=base-value/maxRows*(base-top);svg+=l(left,y,right,y)+t(left-10,y+5,V.num(value),'r-small r-muted','end');}
    current.receiveRows.forEach((count,i)=>{const height=count/maxRows*(base-top);svg+=b(left+i*barWidth+1,base-height,Math.max(1,barWidth-3),height,i===0?'orange':'blue',i===0?.65:.35);});
    const meanY=base-mean/maxRows*(base-top);svg+=l(left,meanY,right,meanY,'faint','stroke-dasharray="4 4"')+t(right,meanY-7,`mean ${V.num(mean)}`,'r-tiny r-muted','end');
    [0,Math.floor((n-1)/2),n-1].forEach(i=>svg+=t(left+(i+.5)*barWidth,278,String(i),'r-small r-muted','middle'));
    svg+=t(left,307,`Rank · max ${V.num(Math.max(...current.receiveRows))} rows`,'r-small')+t(16,49,`${V.num(mean*n)} total routes · ${current.imbalance}× max/mean`,'r-small r-muted');
    const x0=625,x1=1000,maxRate=Math.ceil(Math.max(...rows.map(busy))/1000*1.15),y=value=>base-value/maxRate*(base-top),x=value=>x0+(value-1)/7*(x1-x0);
    for(const value of [0,maxRate/2,maxRate])svg+=l(x0,y(value),x1,y(value))+t(x0-12,y(value)+5,f(value,1),'r-small r-muted','end');
    for(const [get,col]of [[r=>rate(r),'blue'],[busy,'orange']]){const points=rows.map(r=>[x(r.imbalance),y(get(r)/1000)]);svg+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')}" stroke="${c(col)}" class="r-guide"/>`;points.forEach((p,i)=>svg+=dot(p[0],p[1],col,i===phase?8:5));}
    skews.forEach(value=>svg+=t(x(value),278,`${value}×`,'r-small r-muted','middle'));
    svg+=dot(625,302,'blue')+t(640,309,'Cluster mean','r-small')+dot(822,302,'orange')+t(837,309,'Busy rank (derived)','r-small');return svg;
  }
  /** Explain the selected skew while keeping measured and derived rates distinct. */
  function skewCaption(phase){const r=row(S.resultSkewEP,S.resultSkewVariant,'single_layer',3072,skews[phase]);
    if(phase===0)return ['Balanced work distribution',`${V.num(r.receiveRows.reduce((a,b)=>a+b,0))} assignments are split evenly across ${r.ep} GPUs. Cluster mean and derived busiest-rank throughput both equal ${f(rate(r)/1000)} PFLOP/s per GPU.`];
    return [`${skews[phase]}× received-row skew`,`${V.num(Math.max(...r.receiveRows))} rows give the busiest rank a derived ${f(busy(r)/1000,2)} PFLOP/s while cluster mean is ${f(rate(r)/1000,2)}. Maximum-rank TX is ${f(r.nvlinkMaxTXGBps,0)} GB/s (${f(r.nvlinkMaxTXGBps/9,1)}% of one-way theory); the supplied checks pass with ${r.allocationRetries} retries and ${r.ooms} OOMs.`];
  }
  /** Render all raw skew controls with one selected distribution. */
  function renderSkew(phase){variantSelect('skew-variant','resultSkewVariant',S.resultSkewEP);$('skew-EP').value=S.resultSkewEP;const rows=skews.map(i=>row(S.resultSkewEP,S.resultSkewVariant,'single_layer',3072,i));$('skew-config').textContent=`${name(rows[0])} · ${rows[0].mode} · EP${rows[0].ep} · F = 3,072 · total routes fixed · received-row bound = 8× balanced`;
    A.renderSVG('skew-svg',skewSVG(rows,phase));A.renderHTML('skew-table',table(['Max / mean<br>rows','Max received<br>rows','Layer<br>ms','Cluster mean<br>PFLOP/s / GPU','Busy rank, derived<br>PFLOP/s','Max-rank TX<br>GB/s','Retries / OOMs'],rows.map(r=>[`${r.imbalance}×`,V.num(Math.max(...r.receiveRows)),msCell(r),f(rate(r)/1000),f(busy(r)/1000),f(r.nvlinkMaxTXGBps,1),`${r.allocationRetries} / ${r.ooms}`])));tabs('skew-tabs',phase);A.phaseCaption('skew-caption',skewCaption(phase));
  }
  /** Draw proportional feature extents and the three measured width controls. */
  function widthSVG(rows,phase){
    const current=rows[phase];let svg=t(16,25,'Feature widths · one expert','r-head')+t(495,25,'Useful throughput · PFLOP/s / GPU','r-head');
    [['x · D',meta.shape.D,'blue'],['h₁ · 2F',2*current.F,'orange'],['h₂ · F',current.F,'green']].forEach(([label,count,col],i)=>{const y=70+i*69;svg+=t(16,y,`${label} = ${V.num(count)}`)+b(16,y+12,340*count/(2*6144),25,col);});
    svg+=t(16,294,'W₁₃: [2F, D] · W₂: [D, F]','r-small')+t(16,321,'Input/output routing dimensions stay fixed.','r-small r-muted');
    const x0=580,x1=1005,base=250,top=70,maxRate=Math.ceil(Math.max(...rows.map(r=>rate(r,range(r)[0])))/1000*1.18),x=F=>x0+(F-1536)/(6144-1536)*(x1-x0),y=pf=>base-pf/maxRate*(base-top);
    for(const value of [0,maxRate/2,maxRate])svg+=l(x0,y(value),x1,y(value))+t(x0-12,y(value)+5,f(value,1),'r-small r-muted','end');
    const points=rows.map(r=>[x(r.F),y(rate(r)/1000)]),col=colors[current.precision];svg+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')}" stroke="${c(col)}" class="r-guide"/>`;
    rows.forEach((r,i)=>{const [lo,hi]=range(r);svg+=l(x(r.F),y(rate(r,hi)/1000),x(r.F),y(rate(r,lo)/1000),col,'stroke-width="2"')+dot(...points[i],col,i===phase?8:5)+t(x(r.F),279,V.num(r.F),'r-small r-muted','middle');});
    svg+=t(580,309,`F · ${name(current)}`,'r-small')+t(495,342,`${V.num(Math.round(current.forwardFlopsPerLogicalPayloadByteRank0))} FLOPs / logical forward payload byte`,'r-small r-muted');return svg;
  }
  /** Explain measured work growth, not an interpolated latency prediction. */
  function widthCaption(phase){const rows=widths.map(F=>row(S.resultWidthEP,S.resultWidthVariant,'width',F)),r=rows[phase],bf=row(r.ep,'bf16:standard','width',r.F);
    if(phase===2)return ['Amortize routed activation communication over more GEMM work',`Four times the narrow-width GEMM work takes ${f(r.layerMS/rows[0].layerMS,2)}× its latency and delivers ${f(rate(r)/1000)} PFLOP/s per GPU. At F = ${V.num(r.F)}, ${name(r)} is ${f(bf.layerMS/r.layerMS,2)}× the corresponding BF16 width control.`];
    return [`Measured F = ${V.num(r.F)}`,`The input/output feature width remains D = 7,168 while the expert intermediate width is F = ${V.num(r.F)}. ${name(r)} takes ${f(r.layerMS)} ms and delivers ${f(rate(r)/1000)} PFLOP/s at ${V.num(Math.round(r.forwardFlopsPerLogicalPayloadByteRank0))} FLOPs per logical forward payload byte.`];
  }
  /** Render raw width controls, keeping the repeated F3072 control separate. */
  function renderWidth(phase){variantSelect('width-variant','resultWidthVariant',S.resultWidthEP);$('width-EP').value=S.resultWidthEP;const rows=widths.map(F=>row(S.resultWidthEP,S.resultWidthVariant,'width',F));$('width-config').textContent=`${name(rows[0])} · ${rows[0].mode} · EP${rows[0].ep} · balanced routing · D = 7,168 · same logical activation payload`;
    A.renderSVG('width-svg',widthSVG(rows,phase));A.renderHTML('width-table',table(['Expert width<br>F','Layer<br>ms','Useful<br>PFLOP/s / GPU','Dense theory<br>%','Speedup / BF16<br>same width','Mean TX<br>GB/s'],rows.map(r=>[V.num(r.F),msCell(r),f(rate(r)/1000),f(percent(r),1),`${f(row(r.ep,'bf16:standard','width',r.F).layerMS/r.layerMS,2)}×`,f(r.nvlinkTXGBps,1)])));tabs('width-tabs',phase);A.phaseCaption('width-caption',widthCaption(phase));
  }
  /** Model eight identical BF16 layer footprints within one fixed slot. */
  function projection(){const measured=row(64,'bf16:standard'),full=measured.savedBytes[0],input=measured.inputBytes,capacity=measured.slotBytes,records=[];let cursor=0;
    for(let layer=0;layer<8;layer++){const candidate=cursor+full+(7-layer)*input,recompute=candidate>capacity,bytes=recompute?input:full;records.push({layer,offset:cursor,bytes,recompute,candidate});cursor+=bytes;}
    return {measured,full,input,records,cursor,capacity,unused:capacity-cursor,independent:8*measured.arenaBytes,shared:measured.arenaBytes};
  }
  /** Compare separately measured phase windows with actual arena reservations. */
  function policySVG(){const staged=row(64,'mxfp8:staged'),mega=row(64,'mxfp8:mega'),x0=126,x1=490,maxMS=3.1,scale=(x1-x0)/maxMS;let svg=t(16,25,'Measured MXFP8 phase latency','r-head')+t(563,25,'Configured device arena','r-head');
    ['forward','backward','combined'].forEach((phase,i)=>{const y=91+i*63;svg+=t(16,y+5,phase==='combined'?'F + B':phase==='forward'?'Forward':'Backward','r-small');[staged,mega].forEach((r,j)=>{const ms=phase==='combined'?r.layerMS:phase==='forward'?r.forwardMS:r.backwardMS,[lo,hi]=range(r,phase),col=j?'purple':'green',yy=y+(j?10:-9);svg+=whisker(x0+lo*scale,x0+hi*scale,yy,col)+dot(x0+ms*scale,yy,col);});});
    [0,1,2,3].forEach(ms=>svg+=t(x0+ms*scale,271,String(ms),'r-small r-muted','middle'));svg+=t(490,299,'ms','r-small r-muted','end')+dot(26,300,'green')+t(42,307,'Staged','r-small')+dot(180,300,'purple')+t(196,307,'Mega','r-small');
    [staged,mega].forEach((r,i)=>{const y=91+i*110,unit=447/(9*2**30),slotW=r.slotBytes*unit,scratchW=r.scratchBytes*unit;svg+=t(563,y-14,i?'Mega':'Staged')+b(563,y,slotW,39,'green')+b(563+slotW,y,scratchW,39,'amber');if(i)svg+=b(563+(staged.slotBytes+staged.scratchBytes)*unit,y,(mega.scratchBytes-staged.scratchBytes)*unit,39,'purple',.75);svg+=t(563,y+65,`${f(r.slotBytes/2**30,2)} slot + ${f(r.scratchBytes/2**30,2)} scratch = ${f(r.arenaBytes/2**30,2)} GiB`,'r-small');});
    const extra=mega.scratchBytes-staged.scratchBytes,qdata=mega.capacityRows*mega.F,scales=4*mega.capacityRows*Math.ceil(mega.F/32);
    svg+=t(563,301,`Extra h₂ row operand: ${f(extra/2**20,1)} MiB`,'r-label');
    if(extra===qdata+scales)svg+=t(563,330,`${f(qdata/2**20,1)} MiB qdata + ${f(scales/2**20,1)} MiB allocated scales`,'r-small');
    return svg;
  }
  /** Show measured occupied slot bytes within a common arena-size ruler. */
  function occupancySVG(){const rows=[row(64,'bf16:standard'),row(64,'bf16:standard','single_layer',3072,8)],ref=rows[0],unit=1008/ref.arenaBytes,slotW=ref.slotBytes*unit;let svg=t(16,25,'Measured BF16 saved state · same configured arena','r-head')+t(16,54,`Slot: ${V.num(ref.slotBytes/2**20)} MiB`,'r-small')+t(16+slotW+15,54,`Shared scratch: ${V.num(ref.scratchBytes/2**20)} MiB`,'r-small');
    rows.forEach((r,i)=>{const y=96+i*96,used=Math.max(...r.savedBytes),usedW=used*unit;svg+=t(16,y-13,`${r.imbalance}× skew · ${V.num(used/2**20)} MiB saved · max ${V.num(Math.max(...r.receiveRows))} received rows`,'r-small')+b(16,y,slotW,43,'line',.28)+b(16,y,usedW,43,'blue',.6)+b(16+slotW,y,r.scratchBytes*unit,43,'amber')+t(16+slotW+r.scratchBytes*unit/2,y+28,`${V.num(r.scratchBytes/2**20)} MiB scratch reserve`,'r-label','middle');if(slotW-usedW>95)svg+=t(16+usedW+(slotW-usedW)/2,y+28,'Unused slot','r-small r-muted','middle');});
    svg+=l(16+slotW,60,16+slotW,250,'faint','stroke-dasharray="4 4"')+t(16,278,`${V.num(ref.arenaBytes/2**20)} MiB = ${f(ref.arenaBytes/2**30)} GiB planned arena in both cases.`,'r-label')+t(16,313,'All bars use the same byte ruler. PyTorch allocation peaks are shown separately in the table.','r-small r-muted');return svg;
  }
  /** Draw the explicitly projected offsets and retained layer extents. */
  function projectionSVG(){const p=projection(),unit=1008/p.capacity;let svg=t(16,25,'Projection · eight balanced BF16 MoE layers','r-head')+t(16,55,`Fixed ${V.num(p.capacity/2**20)} MiB slot · ${V.num(p.full/2**20)} MiB full state/layer · ${V.num(p.input/2**20)} MiB source input`,'r-small')+b(16,104,1008,57,'line',.25);
    p.records.forEach(rec=>{const x=16+rec.offset*unit,w=rec.bytes*unit;svg+=b(x,104,w,57,rec.recompute?'purple':'green',.4,'stroke-width="2"');if(w>70)svg+=t(x+w/2,129,`L${rec.layer}`,'r-label','middle')+t(x+w/2,151,`${V.num(rec.bytes/2**20)} MiB`,'r-small','middle');svg+=l(x,166,x,185,'green')+t(x+3,207,V.num(rec.offset/2**20),'r-small');});
    const last=p.records.at(-1),mid=16+(last.offset+last.bytes/2)*unit;
    if(last.recompute)svg+=l(mid,132,1024,132,'purple')+l(1024,132,1024,240,'purple');
    svg+=t(16,82,'Each layer starts at the previous cursor.','r-small r-muted')+t(16,238,'Layer start offsets above are MiB from the slot base.','r-small r-muted')+t(1024,263,last.recompute?'L7: save only its input; rebuild expert state in backward.':'All eight layers retain their eligible state.','r-small','end')+t(16,293,`Final cursor: ${V.num(p.cursor/2**20)} MiB · unused slot: ${V.num(p.unused/2**20)} MiB`,'r-label')+t(16,327,`L7 candidate: ${V.num(last.candidate/2**20)} MiB ${last.recompute?'>':'≤'} ${V.num(p.capacity/2**20)} MiB cap → ${last.recompute?'retain input':'keep full state'}.`,'r-small r-muted');return svg;
  }
  /** Keep memory interpretation specific to measured data or the projection. */
  function memoryCaption(phase){const st=row(64,'mxfp8:staged'),me=row(64,'mxfp8:mega');
    if(phase===0)return ['Measured phase performance and scratch tradeoff',`Forward and backward favor different policies here; combined medians differ ${f(100*(me.layerMS/st.layerMS-1),1)}% and their window ranges overlap. Staged reserves ${f((me.scratchBytes-st.scratchBytes)/2**20,0)} MiB less scratch with the same ${f(st.slotBytes/2**30,2)} GiB saved-state slot.`];
    if(phase===1){const bf=row(64,'bf16:standard'),hot=row(64,'bf16:standard','single_layer',3072,8);return ['Measured occupancy follows routing inside a fixed plan',`Saved state grows ${V.num(Math.max(...bf.savedBytes)/2**20)} → ${V.num(Math.max(...hot.savedBytes)/2**20)} MiB and fits the same ${V.num(bf.slotBytes/2**20)} MiB cap. Reported PyTorch allocation peaks are ${f(bf.allocatedPeakGiB,3)} and ${f(hot.allocatedPeakGiB,3)} GiB, with zero retries/OOMs: the tested routing variation fits within fixed reservations.`];}
    const p=projection();return ['Projection: pack layers and choose one replay',`${p.records.filter(r=>!r.recompute).length} full ${V.num(p.full/2**20)} MiB states and ${p.records.filter(r=>r.recompute).length} source-only input fit in the same ${V.num(p.capacity/2**20)} MiB slot. Reusing the scratch arena across eight sequential layers projects ${f(p.independent/2**30,2)} → ${f(p.shared/2**30,2)} GiB of activation-arena reservation; this is not a measured stack.`];
  }
  /** Render memory results and a separately labeled capacity projection. */
  function renderMemory(phase){$('result-memory-config').textContent=phase===0?`Measured · EP64 training · MXFP8 · T = 2,048 · D = 7,168 · F = 3,072 · padded capacity ${V.num(row(64,'mxfp8:mega').capacityRows)} rows`:phase===1?'Measured · EP64 training · BF16 · F = 3,072 · activation/scratch factors 8 · VMM off':'Projection · repeat measured EP64 BF16 footprints · eight layers · fixed slot cap · scratch shared sequentially';
    if(phase===0){const rows=[row(64,'mxfp8:staged'),row(64,'mxfp8:mega')];A.renderSVG('result-memory-svg',policySVG());A.renderHTML('result-memory-table',table(['MXFP8 policy','Forward<br>ms','Backward<br>ms','Layer<br>ms','Slot<br>GiB','Scratch<br>GiB','PyTorch alloc. peak<br>GiB'],rows.map(r=>[r.pipeline==='mega'?'Mega':'Staged',msCell(r,'forward'),msCell(r,'backward'),msCell(r),f(r.slotBytes/2**30,3),f(r.scratchBytes/2**30,3),f(r.allocatedPeakGiB,3)])));}
    else if(phase===1){const rows=[row(64,'bf16:standard'),row(64,'bf16:standard','single_layer',3072,8)];A.renderSVG('result-memory-svg',occupancySVG());A.renderHTML('result-memory-table',table(['BF16 skew','Max saved<br>MiB','Slot cap<br>MiB','Scratch cap<br>MiB','Arena<br>GiB','PyTorch alloc. peak<br>GiB','Retries / OOMs'],rows.map(r=>[`${r.imbalance}×`,V.num(Math.max(...r.savedBytes)/2**20),V.num(r.slotBytes/2**20),V.num(r.scratchBytes/2**20),f(r.arenaBytes/2**30,3),f(r.allocatedPeakGiB,3),`${r.allocationRetries} / ${r.ooms}`])));}
    else{const p=projection();A.renderSVG('result-memory-svg',projectionSVG());A.renderHTML('result-memory-table',table(['Arena projection','Activation reservation<br>MiB','Scratch reservation<br>MiB','Arena reservation<br>MiB'],[['8 separate one-layer contexts',V.num(8*p.measured.slotBytes/2**20),V.num(8*p.measured.scratchBytes/2**20),V.num(p.independent/2**20)],['One shared 8-layer context',V.num(p.capacity/2**20),V.num(p.measured.scratchBytes/2**20),V.num(p.shared/2**20)]],'Projection only · arena reservations; excludes weights, gradients, symmetric buffers, and graph pools'));}
    tabs('result-memory-tabs',phase);A.phaseCaption('result-memory-caption',memoryCaption(phase));
  }
  /** Render the full paired policy table without combining experimental controls. */
  function renderPairs(){const pairs=D.cases.filter(r=>r.precision==='mxfp8'&&r.pipeline==='staged').map(st=>{const me=row(st.ep,'mxfp8:mega',st.category,st.F,st.imbalance),sr=range(st),mr=range(me),interpretation=mr[1]<sr[0]?'Mega ranges lower':sr[1]<mr[0]?'Staged ranges lower':'Window ranges overlap';return [`${st.mode} · EP${st.ep}`,st.category==='width'?'Width control':'Routing control',V.num(st.F),`${st.imbalance}×`,msCell(st),msCell(me),`${f(st.layerMS/me.layerMS,3)}×`,interpretation];});$('result-pairs-table').innerHTML=table(['Workload','Control','F','Skew','Staged ms','Mega ms','Staged / Mega','Observed windows'],pairs);}
  /** Bind manual step tabs to the existing controller implementation. */
  function bindTabs(id,controller){$(id).addEventListener('click',event=>{const button=event.target.closest('[data-step]');if(button)A.controllers[controller].set(Number(button.dataset.step));});}
  A.controllers.efficiency=new A.Controller('efficiency',2,renderEfficiency);
  A.controllers.skew=new A.Controller('skew',3,renderSkew);
  A.controllers.width=new A.Controller('width',3,renderWidth);
  A.controllers['result-memory']=new A.Controller('result-memory',3,renderMemory);
  bindTabs('efficiency-tabs','efficiency');bindTabs('skew-tabs','skew');bindTabs('width-tabs','width');bindTabs('result-memory-tabs','result-memory');
  $('result-mode').addEventListener('click',event=>{const button=event.target.closest('[data-mode]');if(!button)return;S.resultMode=button.dataset.mode;S.resultEP=S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP'];A.controllers.efficiency.render();});
  $('result-EP').addEventListener('change',event=>{S.resultEP=Number(event.target.value);S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP']=S.resultEP;A.controllers.efficiency.render();});
  [['skew-EP','resultSkewEP','skew',true],['skew-variant','resultSkewVariant','skew',false],['width-EP','resultWidthEP','width',true],['width-variant','resultWidthVariant','width',false]].forEach(([id,key,controller,numeric])=>$(id).addEventListener('change',event=>{S[key]=numeric?Number(event.target.value):event.target.value;A.controllers[controller].render();}));
  $('result-presets').addEventListener('click',event=>{const button=event.target.closest('[data-preset]');if(!button)return;S.resultMode=button.dataset.preset;S.resultEP=S[S.resultMode==='inference'?'resultInferenceEP':'resultTrainingEP'];A.controllers.efficiency.render();});
  /** Reset the new figures together with the existing article controls. */
  function reset(){Object.assign(S,defaults);['efficiency','skew','width','result-memory'].forEach(name=>A.controllers[name].set(0));}
  const oldReset=A.reset;A.reset=()=>{oldReset();reset();};$('reset').addEventListener('click',reset);
  const oldCaption=A.getStepCaption;A.getStepCaption=(name,phase)=>name==='efficiency'?efficiencyCaption(phase):name==='skew'?skewCaption(phase):name==='width'?widthCaption(phase):name==='result-memory'?memoryCaption(phase):oldCaption(name,phase);
  /** Sample complete supported modes while presentation reserves a fixed layout. */
  function sample(name,pass){if(name==='efficiency'){for(const mode of ['inference','training'])for(const ep of mode==='inference'?[8,16]:[32,64]){S.resultMode=mode;S.resultEP=ep;pass();}}else if(name==='skew'){for(const ep of [8,16,32,64])for(const variant of variants(ep)){S.resultSkewEP=ep;S.resultSkewVariant=variant;pass();}}else if(name==='width'){for(const ep of [16,64])for(const variant of variants(ep)){S.resultWidthEP=ep;S.resultWidthVariant=variant;pass();}}}
  /** Return a short scope statement matching the current measured/projection view. */
  function scope(name,phase){if(name==='efficiency')return meta.peakNote+(phase===1?' HBM is modeled, not a DRAM counter.':'');if(name==='skew')return 'Busiest-rank throughput is derived from assigned work and maximum-rank latency. NVLink is measured separately; peaks are theoretical.';if(name==='width')return 'Three measured widths; fixed balanced routing and logical activation payload. HBM weight/intermediate traffic grows with F; peaks are theoretical.';return phase===2?'Packing projection, not a measured multi-layer stack. Arena excludes caller weights, gradients, symmetric buffers, and graph pools.':'Measured single-layer EP64 cases. Arena reservations and PyTorch allocation peaks are separate; VMM and capacity recompute are off.';}
  /** Generate the article's numerical prose from the same immutable measurements. */
  function narrative(){
    const bf=row(32,'bf16:standard'),st=row(32,'mxfp8:staged'),balancedNV=row(16,'nvfp4:mega'),hot=row(16,'nvfp4:mega','single_layer',3072,8),small=row(16,'nvfp4:mega','width',1536),large=row(16,'nvfp4:mega','width',6144),ps=row(64,'mxfp8:staged'),pm=row(64,'mxfp8:mega');
    $('result-compute-story').textContent=`For EP32 BF16 training, the complete forward/backward path delivers ${f(rate(bf)/1000)} PFLOP/s per GPU, ${f(percent(bf),1)}% of the dense theoretical Tensor Core peak. MXFP8 Staged delivers ${f(rate(st)/1000)} PFLOP/s. A larger arithmetic peak does not increase NVLink bandwidth or remove the layer’s other work. Read raw throughput, latency, and the hardware reference together.`;
    $('result-skew-story').textContent=`At EP16 with NVFP4, 8× routing skew raises the busiest rank’s work from ${V.num(Math.max(...balancedNV.receiveRows))} to ${V.num(Math.max(...hot.receiveRows))} rows. Its derived useful rate rises from ${f(busy(balancedNV)/1000,2)} to ${f(busy(hot)/1000,2)} PFLOP/s while the cluster average falls because work is concentrated. Maximum-rank TX reaches ${f(hot.nvlinkMaxTXGBps,0)} GB/s, ${f(hot.nvlinkMaxTXGBps/9,1)}% of theoretical one-way NVLink. These observations show useful progress under the tested skew; the router still determines where work is available.`;
    $('result-width-story').textContent=`In the EP16 NVFP4 sweep, quadrupling F creates four times the useful GEMM work while latency grows ${f(large.layerMS/small.layerMS,2)}×, from ${f(small.layerMS)} to ${f(large.layerMS)} ms. Useful throughput reaches ${f(rate(large)/1000)} PFLOP/s per GPU. Speedup over the corresponding BF16 controls grows from ${f(row(16,'bf16:standard','width',1536).layerMS/small.layerMS,2)}× to ${f(row(16,'bf16:standard','width',6144).layerMS/large.layerMS,2)}×. More expert work is amortized over the same logical routed activation payload.`;
    const pairs=D.cases.filter(r=>r.precision==='mxfp8'&&r.pipeline==='staged'),megaWins=pairs.filter(r=>r.layerMS>row(r.ep,'mxfp8:mega',r.category,r.F,r.imbalance).layerMS).length,overlaps=pairs.filter(r=>{const a=range(r),b=range(row(r.ep,'mxfp8:mega',r.category,r.F,r.imbalance));return a[0]<=b[1]&&b[0]<=a[1];}).length;
    $('result-policy-story').textContent=`Staged and Mega retain the same activation-slot budget in the measured EP64 MXFP8 case. Their scratch budgets differ by ${f((pm.scratchBytes-ps.scratchBytes)/2**20,0)} MiB; forward and backward favor different policies, and combined timing ranges overlap. Across ${pairs.length} matched comparisons, ${pairs.length-megaWins} have a lower Staged median and ${megaWins} have a lower Mega median; ${overlaps} pairs have overlapping windows. Dist-MoE provides both policies so the choice can account for shape, phase, and memory capacity.`;
  }
  narrative();renderPairs();window.ResultsApp={data:D,row,work,rate,percent,busy,range,projection,sample,scope,reset};
})();
