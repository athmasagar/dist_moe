'use strict';
// Reflow the existing SVG panels; the figures and numerical data stay shared.
window.ArticleLayout = (() => {
  const layouts=new WeakMap(),wideBounds={'token-svg':'0 0 1600 620','fusion-svg':'0 0 1600 550','skew-svg':'0 0 1600 404','width-svg':'0 0 1600 404'};
  const svgNS='http://www.w3.org/2000/svg';
  function attribute(entry,node,name,value){if(!entry.attrs.has(node))entry.attrs.set(node,new Map());const values=entry.attrs.get(node);if(!values.has(name))values.set(name,node.getAttribute(name));if(value==null)node.removeAttribute(name);else node.setAttribute(name,String(value));}
  function content(entry,node,html){if(!entry.contents.has(node))entry.contents.set(node,node.innerHTML);node.innerHTML=html;}
  function restore(entry){for(const [node,attrs]of entry.attrs)for(const [name,value]of attrs){if(value==null)node.removeAttribute(name);else node.setAttribute(name,value);}for(const [node,html]of entry.contents)node.innerHTML=html;}
  function move(entry,node,x,y){attribute(entry,node,'transform',`translate(${x},${y})`);}
  function textLines(entry,node,x,y,lines,height=29){attribute(entry,node,'x',x);attribute(entry,node,'y',y);content(entry,node,lines.map((line,i)=>`<tspan x="${x}" dy="${i?height:0}">${Views.esc(line)}</tspan>`).join(''));}
  function wrap(entry,node,x,y,width){const words=node.textContent.trim().split(/\s+/),lines=[];let line='';for(const word of words){const candidate=line?line+' '+word:word;node.textContent=candidate;if(node.getComputedTextLength()>width&&line){lines.push(line);line=word;}else line=candidate;}if(line)lines.push(line);textLines(entry,node,x,y,lines);return lines.length;}
  function route(id,d,color='purple'){return `<path id="${id}" d="${d}" fill="none" stroke="var(--${color})" stroke-width="2.2" marker-end="url(#tokens-arrow-${color})"/>`;}
  function routing(svg,entry){
    const shift=590;svg.querySelectorAll('.routing-gpu').forEach(group=>{if(group.dataset.rank==='1')move(entry,group,-868,shift);});
    const fabric=svg.querySelector('.routing-fabric'),stage=Number(fabric.dataset.stage);let html='';
    if(stage===0)html=route('routing-stack-link','M368,526 L368,630')+'<text x="386" y="583" class="small purple-text">Rendezvous</text>';
    if(stage===2)html=route('tokens-path-0','M350,528 L350,630')+route('tokens-path-1','M386,630 L386,528')+'<text x="408" y="567" class="small purple-text">NVLink · IDs</text><text x="408" y="598" class="small green-text">T₀ = T₁</text>';
    if(stage===5){const owner=Number(fabric.dataset.owner),worker=Number(fabric.dataset.worker),token=Number(fabric.dataset.tokenIndex),k=Number(fabric.dataset.k),sy=477+(worker?shift:0),sx=worker?36:700,tx=36+k*332+159,ty=161+token*137+(owner?shift:0),side=worker?8:732;
      html=route('tokens-path-0',`M${sx},${sy} L${side},${sy} L${side},${ty-18} L${tx},${ty-18} L${tx},${ty}`,'blue')+'<text x="230" y="583" class="small blue-text">NVLink · BF16 row · 512 B</text>';
    }
    if(stage===3){const cells=[...fabric.querySelectorAll('.routing-cell')];cells.forEach((cell,i)=>move(entry,cell,-i*533,594+i*82));}
    else content(entry,fabric,html);
    const note=svg.querySelector('.routing-note text');let height=stage===3?1380:1230;
    if(note){if(!entry.contents.has(note))entry.contents.set(note,note.innerHTML);const count=wrap(entry,note,20,1155,700);height=Math.max(height,1168+count*29);}
    svg.setAttribute('viewBox',`0 0 752 ${height}`);
  }
  function fusion(svg,entry,mobile){
    const flow=svg.querySelector('.fusion-flow'),chunks=svg.querySelector('.fusion-chunks'),hardware=svg.querySelector('.fusion-hardware');
    if(!mobile){move(entry,hardware,-800,560);svg.setAttribute('viewBox','0 0 1040 1120');return;}
    const positions=[[20,45],[320,45],[20,200],[320,200],[20,355],[320,355],[20,510],[320,510]];
    [...flow.querySelectorAll('.flow-card')].forEach((group,i)=>{const rect=group.querySelector('rect'),x=Number(rect.getAttribute('x')),y=Number(rect.getAttribute('y'));move(entry,group,positions[i][0]-x,positions[i][1]-y);attribute(entry,rect,'width',260);attribute(entry,rect,'height',78);[...group.querySelectorAll('text')].forEach((node,j)=>attribute(entry,node,'y',y+(j?57:28)));});
    [...flow.querySelectorAll(':scope > rect')].forEach((rect,i)=>{attribute(entry,rect,'width',584);attribute(entry,rect,'y',i?338:8);attribute(entry,rect,'height',i?330:svg.querySelectorAll('.fusion-flow > rect').length===1?660:330);});
    flow.querySelectorAll(':scope > text').forEach(node=>{if(Number(node.getAttribute('x'))>600)attribute(entry,node,'x',582);else if(node.textContent.startsWith('ONE KERNEL GRID'))content(entry,node,'MEGA KERNEL GRID');if(Number(node.getAttribute('y'))===195)attribute(entry,node,'y',348);});
    const paths=[...flow.querySelectorAll(':scope > path')],geometry=['M280,80 L320,80','M450,123 L450,158 L150,158 L150,200','M280,235 L320,235','M450,278 L450,313 L150,313 L150,355','M280,390 L320,390','M450,433 L450,468 L150,468 L150,510','M280,545 L320,545'];
    paths.forEach((path,i)=>attribute(entry,path,'d',geometry[i]));
    svg.querySelectorAll('.edge-payload-label').forEach(group=>{const payload=group.dataset.payload||'',originalY=Number(group.dataset.edgeY),h2=payload.includes('h2'),column=payload.includes('from SMEM')||payload.includes('completed h1')?20:320,y=payload.includes('completed h1')?280:payload.includes('C accumulator')?(originalY>200?595:280):h2?435:125,point=payload.includes('completed h1')?[450,313]:payload.includes('C accumulator')?[300,originalY>200?545:235]:payload.includes('from SMEM')?[450,h2?468:158]:[300,h2?390:80],rect=group.querySelector('rect');
      attribute(entry,group,'data-edge-x',point[0]);attribute(entry,group,'data-edge-y',point[1]);attribute(entry,rect,'x',column);attribute(entry,rect,'y',y);attribute(entry,rect,'height',70);
      const labels=[...group.querySelectorAll('text')];labels.forEach((node,j)=>{attribute(entry,node,'x',column+8);attribute(entry,node,'y',y+26+j*27);});attribute(entry,rect,'width',Math.max(...labels.map(node=>node.getComputedTextLength()))+16);
    });
    // Keep each recorded state, changing only the table's arrangement.
    const labels=[...chunks.querySelectorAll(':scope > text')],rects=[...chunks.querySelectorAll(':scope > rect')];
    labels.forEach(node=>{const y=Number(node.getAttribute('y')),x=Number(node.getAttribute('x'));if(y===353)textLines(entry,node,20,700,['Chunk state · completed / working / waiting']);else if(y===373){const i=Math.round((x-200)/164);attribute(entry,node,'x',110+i*96);attribute(entry,node,'y',733);}else if(y===530)textLines(entry,node,20,908,['Ready counters protect reads.','Consumption credits protect buffer reuse.']);else if(x===20){attribute(entry,node,'y',772+Math.round((y-415)/45)*48);content(entry,node,node.textContent.replace('chunk ','c'));}else{const row=Math.round((y-413)/45),col=Math.round((x-213)/164);attribute(entry,node,'x',115+col*96);attribute(entry,node,'y',772+row*48);}});
    labels.filter(node=>entry.attrs.get(node)?.get('y')==='373').forEach((node,i)=>content(entry,node,['Gather','W13','SwiGLU','W2','Scatter'][i]));
    rects.forEach((rect,i)=>{attribute(entry,rect,'x',105+(i%5)*96);attribute(entry,rect,'y',745+Math.floor(i/5)*48);attribute(entry,rect,'width',90);attribute(entry,rect,'height',40);});
    move(entry,hardware,-1030,960);svg.setAttribute('viewBox','0 0 600 1520');
  }
  function skew(svg,entry){svg.querySelectorAll('.article-wide-title').forEach(node=>attribute(entry,node,'display','none'));svg.querySelectorAll('.skew-distribution').forEach((node,i)=>move(entry,node,-i*342,i*360));move(entry,svg.querySelector('.skew-throughput'),-1104,1130);svg.setAttribute('viewBox','0 0 460 1540');}
  function width(svg,entry){
    const left=svg.querySelector('.width-extents'),right=svg.querySelector('.width-throughput');
    left.querySelectorAll('text').forEach(node=>{const x=Number(node.getAttribute('x')),y=Number(node.getAttribute('y'));if(y===28)content(entry,node,'Expert feature widths');if(y===65)content(entry,node,'x/output: D = 7,168');if(x===195)attribute(entry,node,'x',160);if(y===386)textLines(entry,node,20,386,['h₁: 2F · h₂: F','W₁₃ [2F,D] · W₂ [D,F]']);});
    left.querySelectorAll('.width-feature-strip').forEach(rect=>{attribute(entry,rect,'x',335);attribute(entry,rect,'width',Number(rect.getAttribute('width'))*.7);});
    const heading=right.querySelector('.r-head');if(heading)attribute(entry,heading,'x',940);move(entry,right,-920,455);const footer=right.querySelector('text[y="386"]');if(footer)textLines(entry,footer,940,386,['F · same routed activation payload']);svg.setAttribute('viewBox','0 0 680 885');
  }
  /** Fit panels to the article column while preserving the deck's full diagrams. */
  function apply(svg,forceWide=false){if(!wideBounds[svg.id])return;let entry=layouts.get(svg);if(!entry||entry.identity!==svg.firstChild){entry={identity:svg.firstChild,attrs:new Map(),contents:new Map(),key:''};layouts.set(svg,entry);}
    const deck=forceWide||!!svg.closest('.deck-visual,.deck-print-sheet'),parent=svg.parentElement,style=getComputedStyle(parent),available=parent.clientWidth-parseFloat(style.paddingLeft||0)-parseFloat(style.paddingRight||0),mobile=!deck&&(svg.id==='fusion-svg'?available<800:available<950),key=deck?'deck':`${mobile?'mobile':'article'}-${Math.round(available)}`;
    if(entry.key===key)return;restore(entry);entry.key=key;svg.classList.toggle('article-fit',!deck);parent.classList.toggle('article-responsive',!deck);svg.classList.toggle('article-layout-mobile',mobile);svg.style.removeProperty('max-width');svg.setAttribute('viewBox',wideBounds[svg.id]);
    if(deck)return;
    if(svg.id==='fusion-svg')fusion(svg,entry,mobile);else if(mobile&&svg.id==='token-svg')routing(svg,entry);else if(mobile&&svg.id==='skew-svg'){skew(svg,entry);svg.style.maxWidth='620px';}else if(mobile&&svg.id==='width-svg'){width(svg,entry);svg.style.maxWidth='820px';}
  }
  function tables(root){for(const table of root.querySelectorAll('table')){const headings=[...table.querySelectorAll('thead th')].map(node=>node.textContent.replace(/\s+/g,' ').trim());if(!headings.length)continue;for(const row of table.querySelectorAll('tbody tr'))[...row.children].forEach((cell,i)=>{if(!cell.dataset.label)cell.dataset.label=headings[i]||'';});}}
  return {apply,tables};
})();
