'use strict';
window.Views = (() => {
  const C=n=>`var(--${n})`, esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=n=>n.toLocaleString('en-US'), hex=n=>'0x'+Math.round(n).toString(16).toUpperCase(), bytes=n=>n>=2**30?(n/2**30).toFixed(2)+' GiB':n>=2**20?(n/2**20).toFixed(2)+' MiB':n>=1024?(n/1024).toFixed(2)+' KiB':n+' B';
  const mathHTML=s=>esc(s).replace(/\bB_([MKN])\b/g,'<span class="math-inline">B<sub>$1</sub></span>');
  function mathSVG(s){
    const escaped=esc(s).replace(/\^T/g,'<tspan baseline-shift="super" font-size="78%">T</tspan>');let html='',cursor=0,lowered=false;
    for(const match of escaped.matchAll(/\bB_([MKN])\b/g)){
      html+=`<tspan${lowered?' dy="-3"':''}>${escaped.slice(cursor,match.index)}B</tspan>`;
      html+=`<tspan class="tile-subscript" font-size="78%" dy="3">${match[1]}</tspan>`;
      cursor=match.index+match[0].length;lowered=true;
    }
    if(!lowered)return escaped;
    if(cursor<escaped.length)html+=`<tspan dy="-3">${escaped.slice(cursor)}</tspan>`;
    return html;
  }
  const t=(x,y,s,cls='',anchor='start',extra='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${extra}>${mathSVG(s)}</text>`;
  const equationCache=new Map();
  /** Record shared TeX and its baseline in SVG coordinates. */
  function equation(x,y,tex,cls='',size=26,width=520){
    return `<g class="math-equation ${cls}" data-tex="${esc(tex)}" data-x="${x}" data-baseline="${y}" data-size="${size}" data-width="${width}"></g>`;
  }
  const mathLayers=new WeakMap();
  /** Match the equation layer to its SVG viewport, including scroll and slide scaling. */
  function positionMath(svg){
    const entry=mathLayers.get(svg);if(!entry)return;
    const parent=svg.parentElement,box=svg.getBoundingClientRect(),container=parent.getBoundingClientRect(),view=svg.viewBox.baseVal;
    if(!container.width||!container.height||!view.width)return;
    const sx=parent.clientWidth/container.width,sy=parent.clientHeight/container.height;
    const scale=Math.min(svg.clientWidth/view.width,svg.clientHeight/view.height);
    const left=(box.left-container.left)*sx+parent.scrollLeft+(svg.clientWidth-view.width*scale)/2;
    const top=(box.top-container.top)*sy+parent.scrollTop+(svg.clientHeight-view.height*scale)/2;
    const style=`left:${left}px;top:${top}px;width:${view.width*scale}px;height:${view.height*scale}px;`;
    if(entry.style!==style){entry.layer.style.cssText=style;entry.style=style;}
    const stageStyle=`width:${view.width}px;height:${view.height}px;transform:scale(${scale}) translate(${-view.x}px,${-view.y}px);`;
    if(entry.stageStyle!==stageStyle){entry.stage.style.cssText=stageStyle;entry.stageStyle=stageStyle;}
  }
  /** Typeset each shared equation in an HTML layer that follows the SVG. */
  function renderMath(svg){
    const markers=[...svg.querySelectorAll('.math-equation')];let entry=mathLayers.get(svg);
    if(!markers.length){if(entry){entry.observer.disconnect();entry.layer.remove();mathLayers.delete(svg);}return;}
    if(!entry){
      const layer=document.createElement('div');layer.className='math-layer';const stage=document.createElement('div');stage.className='math-stage';layer.append(stage);svg.after(layer);
      const observer=new ResizeObserver(()=>positionMath(svg));observer.observe(svg);
      entry={layer,stage,observer,markup:''};mathLayers.set(svg,entry);
    }
    const markup=markers.map(marker=>{
      const tex=marker.dataset.tex,size=Number(marker.dataset.size);
      if(!equationCache.has(tex))equationCache.set(tex,{tree:window.katex.__renderToHTMLTree(tex,{throwOnError:true,strict:'error'}),markup:window.katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'})});
      const cached=equationCache.get(tex),top=Number(marker.dataset.baseline)-cached.tree.height*size;
      return `<div class="math-tex ${marker.classList.contains('blue-text')?'math-forward':'math-backward'}" data-tex="${esc(tex)}" data-x="${marker.dataset.x}" data-baseline="${marker.dataset.baseline}" style="left:${marker.dataset.x}px;top:${top}px;font-size:${size}px">${cached.markup}</div>`;
    }).join('');
    if(entry.markup!==markup){entry.stage.innerHTML=markup;entry.markup=markup;}
    positionMath(svg);
  }
  const r=(x,y,w,h,cls='box',extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" class="${cls}" ${extra}/>`;
  const l=(x1,y1,x2,y2,color='line',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C(color)}" stroke-width="1.4" ${extra}/>`;
  let prefix='',motion=[];
  function start(id){prefix=id;motion=[];}
  function wire(d,color='blue',active=true,animate=true,arrow=true,payload='',label=null){const moving=active&&animate&&!!payload,id=`${prefix}-path-${motion.length}`;if(moving)motion.push({id,color,payload,label});return `<path ${moving?`id="${id}" class="flowpath"`:''} d="${d}" fill="none" stroke="${C(active?color:'line')}" stroke-width="${active?2.2:1.2}" ${arrow?`marker-end="url(#${prefix}-arrow-${active?color:'line'})"`:''}/>`;}
  function finish(s){const defs=['blue','orange','green','purple','line','amber'].map(c=>`<marker id="${prefix}-arrow-${c}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8" fill="${C(c)}"/></marker>`).join('');return `<defs>${defs}</defs>${s}${motion.map(edgeLabel).join('')}${motion.map(({id,color,payload})=>`<circle class="packet" data-route="${id}" data-payload="${esc(payload)}" data-color="${color}" cx="0" cy="0" r="5" fill="${C(color)}" visibility="hidden" aria-hidden="true"/>`).join('')}`;}

  // Callouts are attached to the selected edge, independently of its moving dot.
  function edgeLabel({id,color,payload,label}){
    if(!label)return '';
    const {x,y,lines,point}=label,w=Math.max(...lines.map(line=>line.length))*(prefix==='fusion'?11:8)+20,h=lines.length*20+14;
    const lx=Math.max(x,Math.min(x+w,point[0])),ly=Math.max(y,Math.min(y+h,point[1]));
    return `<g class="edge-payload-label" data-route="${id}" data-payload="${esc(payload)}" data-edge-x="${point[0]}" data-edge-y="${point[1]}">`+
      l(lx,ly,point[0],point[1],color,'class="edge-label-leader" stroke-dasharray="2 3"')+
      r(x,y,w,h,'edge-label-bg',`fill="${C('paper')}" stroke="${C(color)}"`)+
      lines.map((line,i)=>t(x+8,y+18+i*20,line,`mono ${color}-text edge-label-text`)).join('')+'</g>';
  }
  function card(x,y,w,h,title,sub,cls='box'){return r(x,y,w,h,cls)+t(x+13,y+24,title,'bold')+(sub?t(x+13,y+46,sub,'small muted'):'');}
  function grid(x,y,w,h,rows=4,cols=8,color='blue',valid=1){let s=r(x,y,w,h,`${color}-fill`);for(let a=0;a<rows;a++)for(let b=0;b<cols;b++){const yy=y+2+a*(h-4)/rows,xx=x+2+b*(w-4)/cols;s+=`<rect x="${xx}" y="${yy}" width="${(w-4)/cols-1}" height="${(h-4)/rows-1}" fill="${C(a/rows<valid?color:'line')}" opacity="${a/rows<valid?.20:.65}"/>`;}if(valid<1)s+=`<rect x="${x+1}" y="${y+h*valid}" width="${w-2}" height="${h*(1-valid)-1}" fill="${C('line')}" opacity=".88"/>`;return s;}
  function badge(x,y,text,color='blue'){return r(x,y,Math.max(55,text.length*7+16),25,`${color}-fill`)+t(x+8,y+17,text,'small');}

  function moe(){start('moe');let s='';
    s+=t(16,23,'ONE TOKEN · TOP-2 ROUTING · FOUR EXPERTS','small muted');
    s+=card(16,203,150,64,'Input token x','One row [D]','blue-fill');
    s+=card(210,55,210,82,'Router','Scores experts for this token','purple-fill')+t(223,122,'Select E0 and E2','small');
    for(let e=0;e<4;e++){const y=48+e*95,selected=e===0||e===2;s+=card(485,y,205,64,`Expert E${e}`,selected?'Selected expert MLP':'Not selected for this token',selected?'green-fill':'box');}
    s+=card(795,203,229,64,'Weighted combine','Σ score × expert(x)','green-fill')+t(808,292,'Output y [D]','small bold');
    s+=wire('M91,203 L91,96 L210,96','blue',true,false)+t(103,182,'Router reads x','small blue-text');
    s+=wire('M166,235 L445,235 L445,96 L485,96','blue',true,false)+wire('M445,235 L445,286 L485,286','blue',true,false)+t(268,258,'Same token row x','small blue-text');
    s+=wire('M420,96 L459,96 L459,70 L485,70','purple',true,false)+wire('M459,96 L459,260 L485,260','purple',true,false)+t(366,165,'Expert IDs','small purple-text');
    s+=wire('M315,55 L315,35 L903,35 L903,203','purple',true,false)+t(768,59,'Route scores','small purple-text');
    s+=wire('M690,80 L748,80 L748,219 L795,219','green',true,false)+wire('M690,270 L748,270 L748,251 L795,251','green',true,false)+t(758,177,'E0(x), E2(x)','small green-text');
    s+=t(16,431,'The caller supplies routing. Dist-MoE executes the selected experts and their distributed dispatch/combine.','small muted');return finish(s);
  }

  function overview(p){start('overview');let s='';
    for(let peer=0;peer<2;peer++){const y=40+peer*290;s+=r(12,y,205,265,'panel')+t(27,y+26,`Peer GPU ${peer+1}`,'bold large')+t(27,y+46,'token owner · device HBM','small muted');s+=card(28,y+60,172,82,'Dispatch buffer','BF16 x (peer HBM)',p<=1?'blue-fill':'box')+grid(41,y+120,144,18,1,8,'blue');s+=card(28,y+162,172,62,'Combine buffer','BF16 routes (peer HBM)',p>=7?'green-fill':'box');s+=t(28,y+246,p>=8?'output [T,D] → local HBM':'publish → GPU barrier','small '+(p===0?'purple-text':'muted'));}
    s+=r(305,40,720,625,'panel')+t(324,68,'Current GPU · expert owner','large bold')+r(324,82,680,283,'green-fill','stroke-opacity=".5"')+t(342,108,'LOCAL HBM · weights + activation-buffer views','small green-text');
    s+=card(345,130,194,78,'x_gathered','BF16 [R, D]',p===1||p===2?'blue-fill':'box')+t(358,194,'base + o_xg','mono muted');
    s+=card(780,130,204,78,'h1 · gate / up','BF16 [R, 2F]',p===4||p===5?'green-fill':'box')+t(794,194,'base + o_h1','mono muted');
    s+=card(345,262,194,77,p>=6?'W2 weights':'W13 weights','caller-owned local HBM','box');
    s+=card(560,262,188,77,'h2','BF16 [R, F]',p===5||p===6?'green-fill':'box')+t(573,326,'base + o_h2','mono muted');
    s+=r(324,384,680,258,'orange-fill','stroke-opacity=".4"')+t(342,410,'CTA · compute pipeline','small orange-text');
    s+=card(780,425,204,63,'SwiGLU registers','FP32 nonlinear arithmetic',p===5?'orange-fill':'box');
    s+=card(345,515,185,105,'TMA → SMEM',p>=6?'A = h2 · B = W2':'A = x · B = W13',p===2||p===6?'blue-fill':'box');s+=grid(360,573,72,29,2,6,'blue')+grid(444,573,68,29,2,6,'green');
    s+=card(570,515,184,105,'Tensor cores',p>=6?'W2 · FP32 TMEM':'W13 · FP32 TMEM',p===3||p===6?'orange-fill':'box')+grid(585,573,152,29,2,8,'orange');
    s+=card(790,515,194,105,'Epilogue','TMEM → registers → SMEM',p===4||p===7?'green-fill':'box')+t(805,599,'convert result → BF16','small');
    s+=wire('M200,132 C256,132 257,167 345,167','blue',p===1,true,true,'source row [D] · BF16',{x:230,y:80,lines:['x [D]','BF16'],point:[259,149]})+wire('M200,422 C263,422 267,178 345,178','blue',p===1,true,true,'source row [D] · BF16',{x:215,y:244,lines:['x [D]','BF16'],point:[278,264]});
    s+=wire('M440,208 L440,210 L330,210 L330,492 L410,492 L410,515','blue',p===2,true,true,'activation A tile [B_M,B_K] · BF16',{x:350,y:214,lines:['A [B_M,B_K]','BF16 → SMEM'],point:[330,236]})+wire('M539,300 L548,300 L548,501 L490,501 L490,515','green',p===2||p===6,true,true,'weight B tile [B_K,B_N] · BF16',{x:350,y:430,lines:['B [B_K,B_N]','BF16 → SMEM'],point:[548,452]});
    s+=wire('M530,555 L570,555','orange',p===3||p===6,true,true,'A/B tiles from SMEM → MMA',{x:568,y:450,lines:['A/B operand tiles','SMEM → MMA'],point:[550,555]})+wire('M754,555 L790,555','orange',p===4||p===7,true,true,'C accumulator [B_M,B_N] · FP32',{x:588,y:433,lines:['C [B_M,B_N]','FP32'],point:[772,555]});
    s+=wire('M886,515 L886,505 L997,505 L997,218 L886,218 L886,208','green',p===4,true,true,'h1 output tile · BF16',{x:784,y:262,lines:['h1 [B_M,B_N]','BF16 → HBM'],point:[997,284]})+wire('M926,208 L926,425','orange',p===5,true,true,'h1 gate/up values [2F] · BF16',{x:760,y:245,lines:['h1 [2F] · BF16','→ SwiGLU'],point:[926,267]})+wire('M780,455 C749,455 659,419 659,339','green',p===5,true,true,'h2 activation values [F] · BF16',{x:700,y:355,lines:['h2 [F] · BF16','→ local HBM'],point:[659,386]});
    s+=wire('M610,339 C610,478 430,450 430,515','blue',p===6,true,true,'h2 activation A tile · BF16',{x:628,y:368,lines:['h2 A tile · BF16','→ SMEM'],point:[610,390]});
    s+=wire('M790,598 L774,598 L774,652 L244,652 L244,237 L200,237','blue',p===7,true,true,'expert result row [D] · BF16',{x:171,y:305,lines:['h3[D]','BF16'],point:[244,327]})+wire('M790,606 L782,606 L782,660 L276,660 L276,527 L200,527','blue',p===7,true,true,'expert result row [D] · BF16',{x:281,y:575,lines:['h3[D]','BF16'],point:[276,597]});
    s+=t(258,24,'NVLink','small blue-text','middle')+t(258,42,'fabric','small muted','middle');
    s+=t(324,693,'Source publication and final peer stores are ordered by GPU symmetric-memory barriers.','small muted');return finish(s);
  }
  const tokenPhases=['Peer mappings','Publish','Exchange IDs','Build pointers','Gather','Scatter','Reduce'];
  /** Follow publication, local metadata construction, and pointer-driven payload movement. */
  function tokens(step,focus='all'){
    start('tokens');const data=Moe.tokenData,chosen=focus==='all'?'C':focus,v=data[chosen],expert=v.experts.find(e=>Math.floor(e/2)!==v.rank),k=v.experts.indexOf(expert),worker=Math.floor(expert/2);let s='';
    const titles=['Peer mappings · once per context','Publish IDs and source rows','Exchange IDs → count expert rows','Build addresses from rank, token, and k','Gather → expert-major rows','Scatter → original top-k slots','Weight → sum → return'];
    const symbol=(base,rank)=>base+(rank?'₁':'₀');
    const cell=(x,y,w,h,label,value,color='box',extra='')=>`<g class="routing-cell">`+r(x,y,w,h,color,extra)+t(x+w/2,y+23,label,'small bold','middle')+(value!==''?t(x+w/2,y+50,value,'mono','middle'):'')+'</g>';
    const ids=(rank,x,y,filter=false)=>{let out='';for(const [i,name]of ['A','B','C','D'].entries()){
      out+=t(x,y+i*43+26,`${name} / r${data[name].rank}`,'small');data[name].experts.forEach((id,slot)=>{const active=!filter||Math.floor(id/2)===rank,xx=x+126+slot*251;out+=cell(xx,y+i*43,235,38,`E${id}`, '',active?'purple-fill':'box',active?'':'opacity=".3"');});}return out;};
    s+=t(20,31,titles[step],'bold');
    for(let rank=0;rank<2;rank++){
      const x=rank?884:16,local=rank?['C','D']:['A','B'],rows=Moe.routingRows(rank);
      s+=`<g class="routing-gpu" data-rank="${rank}">`+r(x,55,704,465,'panel')+t(x+20,91,`GPU ${rank} · E${rank*2} / E${rank*2+1}`,'large bold');
      if(step===0){
        [['Routing','int16 [2,2] · header 128 B','ρ','purple'],['Dispatch','BF16 [2,2,256] capacity','δ','blue'],['Combine','BF16 [2,2,256]','κ','green']].forEach(([name,shape,base,color],i)=>{
          const yy=124+i*108;s+=t(x+20,yy,`${name} · ${symbol(base,rank)}`,'small bold')+t(x+684,yy,shape,'small muted','end');
          for(let j=0;j<4;j++)s+=cell(x+20+j*166,yy+13,156,60,name==='Routing'?`k${j%2}`:name==='Dispatch'?`row ${j}`:`t${Math.floor(j/2)} / k${j%2}`,'',`${color}-fill`);
        });
        [['ρ₀,ρ₁','Routing'],['δ₀,δ₁','Dispatch'],['κ₀,κ₁','Combine']].forEach(([value,label],i)=>s+=cell(x+20+i*225,449,214,58,label,`[${value}]`,'box'));
      }
      if(step===1){
        s+=t(x+20,126,'Caller IDs / local scores','small bold');
        local.forEach((name,i)=>data[name].experts.forEach((e,slot)=>{const j=i*2+slot;s+=cell(x+20+j*166,139,156,77,`${name}/k${slot} → E${e}`,`score ${data[name].scores[slot]}`,'purple-fill');}));
        s+=wire(`M${x+352},216 L${x+352},250`,'purple',true,true,true,'int16 IDs + int32 T')+t(x+385,240,'IDs','small purple-text');
        s+=cell(x+20,253,664,60,`Routing ${symbol('ρ',rank)} · T = 2`,local.map(n=>`${n}:[${data[n].experts.join(',')}]`).join('  '),'purple-fill');
        s+=t(x+20,348,`x → dispatch ${symbol('δ',rank)} · BF16 [2,256]`,'small bold');
        local.forEach((name,i)=>{s+=cell(x+20+i*332,364,318,66,`${name} · ${i*512} B`,'256 values','blue-fill')+grid(x+35+i*332,434,288,16,1,8,'blue');});
        s+=t(x+352,488,'2 × 512 B = 1,024 B · one copy per token','small muted','middle');
      }
      if(step===2){
        s+=cell(x+20,119,664,61,`Routing ${symbol('ρ',rank)} · int32 T = 2`,local.map(n=>`${n}:[${data[n].experts.join(',')}]`).join('  '),'purple-fill');
        s+=t(x+20,239,'Local int16 [EP,T,K] · 16 B','small bold')+t(x+684,239,'k0       k1','mono','end');
        s+=ids(rank,x+20,256,true);
        for(let e=rank*2;e<rank*2+2;e++){const names=rows.filter(r=>r.expert===e).map(r=>r.name),xx=x+20+(e%2)*332;s+=cell(xx,441,318,64,`E${e} · count 2`,`${names.join(',')} · prefix ${(e%2)*2}`,'green-fill');}
      }
      if(step===3){
        s+=t(x+20,129,'Route','small bold')+t(x+190,129,'Read δ','small blue-text')+t(x+420,129,'Write κ','small green-text');
        rows.forEach((r0,i)=>{const yy=148+i*64,active=r0.name===chosen;s+=`<g class="routing-cell">`+r(x+20,yy,664,52,active?'amber-fill':'box')+t(x+32,yy+34,`E${r0.expert}(${r0.name}) / k${r0.k}`,'mono')+t(x+190,yy+34,`${symbol('δ',r0.owner)}+${r0.gatherByte}B`,'mono blue-text')+t(x+420,yy+34,`${symbol('κ',r0.owner)}+${r0.returnByte}B`,'mono green-text')+'</g>';});
        s+=t(x+20,451,'Local int64 pointer tables','small muted')+t(x+20,489,'Counts [2,2] → row starts [0,2]','mono');
      }
      if(step===4){
        s+=t(x+20,130,'Dispatch','small blue-text')+t(x+390,130,'x_gathered · BF16 [4,256]','small green-text');
        rows.forEach((r0,i)=>{const yy=150+i*71,col=r0.owner===rank?'green':'blue';
          s+=cell(x+20,yy,270,61,`${r0.name} · ${symbol('δ',r0.owner)}+${r0.gatherByte}B`,'','blue-fill')+grid(x+33,yy+36,244,15,1,8,'blue');
          s+=cell(x+390,yy,294,61,`E${r0.expert} / ${r0.name}`,`o_xg+${i*512}B`,'green-fill')+wire(`M${x+290},${yy+31} L${x+390},${yy+31}`,col,true,true,true,`${r0.name} row · BF16 [256]`)+t(x+340,yy+16,'512 B','small','middle');
        });
        ['W13','SwiGLU','W2'].forEach((label,i)=>{s+=cell(x+20+i*230,455,204,48,label,'','orange-fill');if(i<2)s+=wire(`M${x+224+i*230},479 L${x+250+i*230},479`,'orange',true,false);});
      }
      if(step===5){
        s+=t(x+20,128,`Combine ${symbol('κ',rank)} · BF16 [T,K,D]`,'small green-text');
        local.forEach((name,i)=>data[name].experts.forEach((e,slot)=>{const xx=x+20+slot*332,yy=161+i*137,offset=(i*2+slot)*512,active=name===chosen&&slot===k;
          s+=cell(xx,yy,318,114,`${name} / k${slot} ← E${e}(${name})`,`${symbol('κ',rank)}+${offset}B`,active?'amber-fill':'green-fill')+grid(xx+13,yy+67,292,20,1,8,'green')+t(xx+159,yy+105,`${Math.floor(e/2)===rank?'local':'NVLink'} · GPU ${Math.floor(e/2)}`,'small muted','middle');
        }));
        if(rank===worker)s+=cell(x+(worker?20:404),446,280,61,`W2 · E${expert}(${chosen})`,'scatter_ptr','orange-fill');
      }
      if(step===6){
        local.forEach((name,i)=>{const q=data[name],yy=158+i*169,out=q.outputs[0].map((n,j)=>Number((n*q.scores[0]+q.outputs[1][j]*q.scores[1]).toFixed(2)));
          s+=t(x+20,yy-32,name,'bold');q.outputs.forEach((values,slot)=>{const xx=x+20+slot*218;s+=t(xx+96,yy-11,`× ${q.scores[slot]}`,'small purple-text','middle')+cell(xx,yy,192,78,`k${slot} / E${q.experts[slot]}`,`[${values.join(',')}]`,'green-fill');});
          s+=wire(`M${x+212},${yy+39} L${x+222},${yy+39} L${x+222},${yy+96} L${x+459},${yy+96} L${x+459},${yy+48}`,'orange',true,false)+wire(`M${x+430},${yy+57} L${x+459},${yy+57}`,'orange',true,false)+`<circle cx="${x+480}" cy="${yy+48}" r="23" fill="var(--orange-soft)" stroke="var(--orange)"/>`+t(x+480,yy+58,'Σ','large bold','middle');
          s+=wire(`M${x+503},${yy+48} L${x+535},${yy+48}`,'orange',true,false)+cell(x+535,yy,149,78,'FP32',`[${out.join(',')}]`,'orange-fill')+wire(`M${x+610},${yy+78} L${x+610},${yy+100}`,'green',true,false)+t(x+610,yy+124,`BF16 y_${name}`,'small green-text','middle');
        });
      }
      s+='</g>';
    }
    s+=`<g class="routing-fabric" data-stage="${step}" data-owner="${v.rank}" data-worker="${worker}" data-token-index="${v.index}" data-k="${k}">`;
    if(step===0)s+=wire('M720,302 L884,302','purple',true,false)+t(802,278,'Rendezvous','small purple-text','middle');
    if(step===2)s+=wire('M700,146 L770,146 L770,279 L904,279','purple',true,true,true,'expert IDs · int16')+wire('M904,166 L834,166 L834,366 L700,366','purple',true,true,true,'expert IDs · int16')+r(754,194,95,42,'edge-label-bg','fill="var(--paper)" stroke="var(--purple)"')+t(802,221,'IDs','small purple-text','middle')+t(802,452,'T₀ = T₁','small green-text','middle');
    if(step===3){const tuple=`r${v.rank} / t${v.index} / k${k}`,offset=(v.index*2+k)*512;[['Gather',`${symbol('δ',v.rank)} + ${v.index}×512 = ${symbol('δ',v.rank)}+${v.index*512}B`,'blue'],['Scatter',`${symbol('κ',v.rank)} + (${v.index}×2+${k})×512 = ${symbol('κ',v.rank)}+${offset}B`,'green'],['Backward return',`${symbol('δ',v.rank)} + (${v.index}×2+${k})×512 = ${symbol('δ',v.rank)}+${offset}B`,'purple']].forEach(([label,value,color],i)=>s+=cell(16+i*533,541,516,67,`${label} · ${tuple}`,value,`${color}-fill`));}
    if(step===5){const ownerX=v.rank?884:16,sourceX=worker?904:700,sourceY=477,targetX=ownerX+20+k*332+159,targetY=161+v.index*137;s+=wire(`M${sourceX},${sourceY} L802,${sourceY} L802,${targetY-18} L${targetX},${targetY-18} L${targetX},${targetY}`,'blue',true,true,true,`${chosen}: h3 BF16 [256] · 512 B`)+r(750,382,104,55,'edge-label-bg','fill="var(--paper)" stroke="var(--blue)"')+t(802,405,'BF16','small blue-text','middle')+t(802,428,'512 B','small blue-text','middle');}
    s+='</g>';
    if(step!==3)s+='<g class="routing-note">'+t(20,584,['int64 peer bases · physical HBM stays on its GPU','IDs: int16 · token count: int32 · scores stay local','EP publication barrier precedes peer reads','', 'Blue: NVLink read · green: local read','EP combine barrier precedes owner reads','Two components shown · score weighting and sum are illustrated before BF16 rounding'][step],'small muted')+'</g>';
    return finish(s);
  }
  function consumption(c,plan){start('consumption');const total=plan.virtual,act=plan.activation,host=plan.hostScratch,hot=plan.deviceScratch,scale=1008/total;let s=t(16,22,c.vmm?`VMM on: one GPU virtual range · ${host?'HBM + pinned host backing':'device HBM backing'}`:'VMM off: one GPU allocation · device HBM only','bold')+t(1024,45,`Combined allocation: ${bytes(total)}`,'small muted','end');
    const regions=[{name:'Activation slots',value:act,color:'green',location:'Device HBM',detail:`${c.N} × ${bytes(plan.slot)} per slot`},{name:'Host overflow scratch',value:host,color:'purple',location:c.vmm?'Pinned host physical pages':'VMM disabled',detail:'Scratch beyond the hot device tier'},{name:'Hot shared scratch',value:hot,color:'amber',location:'Device HBM',detail:'Reused across sequential local calls'}];
    let x=16;for(const region of regions){const width=region.value*scale;if(width>0)s+=`<g><title>${esc(region.name+': '+num(region.value)+' bytes')}</title>`+r(x,65,width,42,`${region.color}-fill`,'rx="0"')+(width>180?t(x+width/2,92,region.name,'small bold','middle'):'')+'</g>';x+=width;}
    s+=r(16,65,1008,42,'','fill="none" stroke="var(--line)"')+t(16,127,'saved slots grow →','small green-text')+t(1024,127,'← scratch grows from the high end','small amber-text','end');
    regions.forEach((region,i)=>{const xx=16+i*344;s+=r(xx,153,320,103,`${region.color}-fill`)+t(xx+13,177,region.name,'bold')+t(xx+13,201,bytes(region.value),'large bold')+t(xx+13,224,region.location,'small')+t(xx+13,244,region.detail,'tiny muted');});
    s+=t(16,280,'Segment widths are proportions of the current combined allocation; the byte totals remain explicit.','small muted');return finish(s);
  }
  function calculations(c,plan,view){start('memory');const saved=view==='saved',forward=view==='forward',rows=saved?plan.rb:view==='total'?plan.rt:plan.rd,parts=saved?plan.savedParts:forward?Moe.forwardScratchParts(c,rows):view==='device'?plan.deviceParts:plan.totalParts,total=Moe.sum(parts);let s=t(16,22,saved?'Saved state for one layer':forward?'Forward scratch · training with eligible state omitted':'Backward scratch · recompute peak','bold');
    s+=r(16,40,1008,47,'blue-fill')+t(30,69,`R = ${num(rows)} · D = ${num(c.D)} · F = ${num(c.F)} · T = ${num(c.T)} · K = ${c.K}`,'mono');
    const columns=parts.length<=4?parts.length:4,w=(1008-(columns-1)*16)/columns,h=108;
    parts.forEach((p,i)=>{const x=16+(i%columns)*(w+16),y=110+Math.floor(i/columns)*124;const color=p.dtype==='BF16'?'blue':p.dtype.includes('E8M0')?'purple':'green';const limit=Math.floor((w-24)/7.4);let formula;if(p.name==='Shared row-operand scratch'||p.name==='Forward row operands')formula=forward?[c.pipeline==='mega'?'row x + row h2':'max(row x, row h2)']:c.pipeline==='mega'?['max(row x + row h2,','row grad_h1)']:['max(row x, row h2,','row grad_h1)'];else formula=p.formula.length>limit?[p.formula.slice(0,limit),p.formula.slice(limit)]:[p.formula];s+=r(x,y,w,h,`${color}-fill`)+t(x+12,y+23,p.name,'small bold')+formula.map((f,j)=>t(x+12,y+45+j*16,f,'tiny mono')).join('')+t(x+12,y+85,bytes(p.bytes),'large bold');});
    const yy=parts.length>4?357:250;s+=r(16,yy-7,1008,37,'amber-fill')+t(30,yy+17,saved?`Σ tensor bytes = ${num(total)} B per layer × ${c.L} layers = ${bytes(total*c.L)} balanced full-save`:`Σ live regions = ${num(total)} B = ${bytes(total)} ${forward?'forward':view==='device'?'backward':'total'} scratch`,'small bold');if(!saved&&c.precision==='bf16'&&c.D<=c.F)s+=t(16,316,forward?'h3 goes to symmetric combine HBM; source x is retained in the activation slot.':'h2’s last read finishes before grad_h1 reuses its region; count the larger extent once.','small muted')+t(16,351,forward?'Forward: 2R(D + 3F), with align128 applied to each region.':'Backward: 2R(D + 5F), with align128 applied to each region.','mono')+t(16,378,'Same received-row bound R in both views · sequential layers share this scratch.','small muted');return {svg:finish(s),parts,rows,total};
  }
  function packing(plan,phase) {
    start('packing');const snap=plan.snapshots[phase],a=snap.record,scale=1008/plan.slotBytes;
    const colors=['blue','green','purple','orange'];let s='';
    s+=t(16,23,'FOUR MoE LAYERS · SAME SLOT OWNER','small bold')+t(1024,23,phase===0?'Before forward':`${snap.kind==='forward'?'After forward':'After backward'} L${a.layer}`,'bold','end');
    plan.records.forEach((rec,i)=>{
      const x=16+i*256,live=snap.live.includes(rec),active=a===rec,y=39;
      s+=r(x,y,240,78,active?`${colors[i]}-fill`:'box')+t(x+12,y+22,`L${i} · rows [${rec.rows.join(', ')}]`,'small bold');
      s+=t(x+12,y+43,phase===0||snap.kind==='forward'&&i>a.layer?'Not yet planned':live?rec.recompute?'Keep x · replay later':'Keep x_gathered / h1 / h3':'Released by backward','small');
      s+=t(x+12,y+64,live?`${num(rec.end-rec.base)} B on rank ${plan.rank}`:active?'Released; cursor rewound':'','mono muted');
    });
    s+=t(16,143,'ONE RANK-LOCAL ACTIVATION ALLOCATION · DEVICE HBM','small bold');
    s+=r(16,156,260,44,'box')+t(146,184,'Slot 0 · other owner','small muted','middle');
    s+=r(284,156,452,44,'green-fill')+t(510,184,`Slot 1 · ${bytes(plan.slotBytes)} capacity`,'small bold','middle');
    s+=r(744,156,280,44,'amber-fill')+t(884,184,`Shared scratch · ${bytes(plan.scratch)}`,'small','middle');
    s+=l(284,201,16,239,'green','stroke-dasharray="3 4"')+l(736,201,1024,239,'green','stroke-dasharray="3 4"');
    s+=t(16,228,`SLOT 1 DETAIL · base-relative byte interval [${hex(plan.start)}, ${hex(2*plan.slotBytes)})`,'small bold')+t(1024,228,'xg = x_gathered','tiny muted','end');
    s+=r(16,242,1008,55,'panel');
    for(const rec of snap.live) {
      const x=16+(rec.base-plan.start)*scale,w=(rec.end-rec.base)*scale;
      s+=r(x,242,w,55,`${colors[rec.layer]}-fill`)+t(x+w/2,274,`L${rec.layer}`,'small bold','middle');
      if(!rec.recompute&&w>150){let off=x;rec.items.forEach(item=>{const ww=item.bytes*scale;if(off>x)s+=l(off,246,off,293,colors[rec.layer]);if(ww>40)s+=t(off+ww/2,291,item.name==='x_gathered'?'xg':item.name,'tiny','middle');off+=ww;});}
    }
    const used=snap.cursor-plan.start,reserve=snap.kind==='forward'?a.reserve:0,free=plan.slotBytes-used;
    if(reserve) s+=r(16+used*scale,242,reserve*scale,55,'','fill="none" stroke="var(--amber)" stroke-dasharray="4 3"');
    if(snap.kind==='backward')s+=r(16+(a.base-plan.start)*scale,242,(a.end-a.base)*scale,55,'','fill="none" stroke="var(--line)" stroke-dasharray="4 3"');
    const cursorX=16+used*scale;
    s+=l(cursorX,235,cursorX,305,'green','stroke-width="2.5"')+t(16,324,`buffer_offsets[1] = ${hex(snap.cursor)} · ${num(used)} B saved`,'mono green-text');
    s+=t(1024,324,reserve?`${num(reserve)} B input reserve (dashed) · ${num(free-reserve)} B uncommitted`:`${num(free)} B free in this slot`,'small muted','end');
    s+=t(16,355,'SCRATCH DETAIL · TEMPORARY VIEWS CARVED FROM THE FIXED HIGH END','small bold');
    s+=r(16,374,1008,48,'panel');
    let temporary=[];
    if(a&&snap.kind==='forward')temporary=a.forwardScratch;
    else if(a)temporary=a.backwardScratch;
    const scratchScale=1008/plan.scratch;
    temporary.forEach((item,i)=>{const x=16+(item.offset-2*plan.slotBytes)*scratchScale,w=item.bytes*scratchScale;s+=r(x,374,w,48,'amber-fill');if(w>38){const label=item.name.includes('grad_h1')?'reuse':item.name.includes('x_gathered')?(w>130?item.name:'xg'):item.name.includes('Replayed')?'h1':item.name;s+=t(x+w/2,402,label,'small','middle');}});
    s+=t(16,443,a?`This call: ${temporary.map(p=>p.name.replace('Replayed ','')).join(' · ')} · ${num(temporary.reduce((n,p)=>n+p.bytes,0))} B peak footprint`:'No temporary views before forward. Scratch has one fixed high end.','small muted');
    s+=r(16,461,598,192,a?.recompute?'orange-fill':'green-fill')+r(632,461,392,192,'box');
    if(!a){
      s+=t(30,488,'DEVICE LOOK-AHEAD','small bold')+t(30,520,`Mandatory budget: 4 inputs × ${num(plan.input)} B = ${num(plan.minimum)} B`,'small');
      s+=t(30,547,'Advance to append L0. Change capacity to alter later decisions.','small')+t(30,575,'The rank selector changes the inspected bytes, not the decision.','small');
      s+=t(646,488,'PLANNER OFFSETS','small bold')+t(646,521,`buffer_offsets[1] = ${hex(plan.start)}`,'mono')+t(646,550,`buffer_offsets[N] = ${hex(plan.scratchEnd)}`,'mono')+t(646,580,'Address = buffer_base + byte offset','small')+t(646,608,'Each forward saves its own offset record.','small');
    }else if(snap.kind==='forward'){
      s+=t(30,488,`L${a.layer} LOOK-AHEAD · SAME DECISION ON BOTH RANKS`,'small bold');
      for(let rank=0;rank<2;rank++)s+=t(30,519+rank*27,`rank ${rank}: ${num(a.before[rank])} saved + ${num(a.full[rank])} full = ${num(a.before[rank]+a.full[rank])} B`,'mono');
      s+=t(30,579,`max + ${plan.depth-a.layer-1} × ${num(plan.input)} = ${num(a.test)} B ${a.recompute?'>':'≤'} ${num(plan.slotBytes)} B`,'mono bold');
      s+=t(30,611,`need_recompute = ${a.recompute?1:0} → ${a.recompute?'append only source x':'append full eligible state'}`,'small bold');
      s+=t(646,488,`L${a.layer} SAVED VIEWS · EP RANK ${plan.rank}`,'small bold');
      a.items.forEach((item,i)=>s+=t(646,519+i*27,`${item.name}: ${hex(item.offset)} + ${num(item.bytes)} B`,'mono'));
      s+=t(646,621,'Address = buffer_base + byte offset','small');
    }else{
      s+=t(30,488,`BACKWARD L${a.layer} · ${a.recompute?'REPLAY SAVED x':'READ SAVED STATE'}`,'small bold');
      s+=t(30,521,a.recompute?'Rebuild expert intermediates in shared scratch.':'Use retained x_gathered / h1 / h3; rebuild h2.','small');
      s+=t(30,550,'Gradients consume these values before the saved region is freed.','small');
      s+=t(30,579,`Rewind cursor: ${hex(a.end)} → ${hex(a.base)}`,'mono bold');
      s+=t(30,611,`Saved bytes after release: rank 0 = ${num(snap.totals[0])}, rank 1 = ${num(snap.totals[1])}`,'small');
      s+=t(646,488,'USE THE MATCHING FORWARD RECORD','small bold')+t(646,521,`Saved physical slot ID = 1`,'mono');
      s+=t(646,551,`Restore its start = ${hex(a.base)}`,'mono')+t(646,581,`moe_layer_id[1] = ${snap.live.length}`,'mono')+t(646,621,'Later layers are released first (LIFO).','small');
    }
    s+=t(16,677,'Solid layer regions: saved bytes · dashed amber: reserved input budget · dashed gray: just released · no host backing here','small muted');
    return finish(s);
  }
  function lookahead(plan,phase){
    start('lookahead');const snap=plan.snapshots[phase],rec=snap.record||plan.records[0],backward=snap.kind==='backward';
    const totals=rec.before.map((n,r)=>n+rec.full[r]+rec.reserve),extent=Math.max(plan.slotBytes,...totals)*1.05,unit=980/extent,capX=30+plan.slotBytes*unit;
    let s=t(16,24,backward?`BACKWARD L${rec.layer} USES ITS EARLIER FORWARD CHOICE`:phase===0?'PREVIEW THE L0 FORWARD CAPACITY TEST':`FORWARD L${rec.layer} · TEST HYPOTHETICAL FULL STATE`,'bold');
    s+=r(16,44,12,12,'green-fill')+t(35,56,'Already saved','small')+r(223,44,12,12,'blue-fill')+t(242,56,'This layer’s full-state candidate','small');
    s+=r(535,44,12,12,'amber-fill','stroke-dasharray="3 2"')+t(554,56,'Future input reserve','small')+l(806,44,806,59,'orange','stroke-width="2.5"')+t(820,56,`Capacity ${num(plan.slotBytes)} B`,'small');
    s+=r(16,80,1008,221,'panel');
    for(let rank=0;rank<2;rank++){
      const y=121+rank*104,before=rec.before[rank],full=rec.full[rank],total=totals[rank],overflow=total>plan.slotBytes;
      s+=t(30,y-17,`Rank ${rank} (R=${rec.rows[rank]}): ${num(before)} saved + ${num(full)} candidate + ${num(rec.reserve)} reserve = ${num(total)} B`,'mono');
      let x=30;for(const [amount,color]of [[before,'green'],[full,'blue'],[rec.reserve,'amber']]){if(amount)s+=r(x,y,amount*unit,38,`${color}-fill`,color==='amber'?'stroke-dasharray="4 3"':'');x+=amount*unit;}
      s+=l(capX,y-6,capX,y+45,'orange','stroke-width="2.5"');
      s+=t(30,y+58,overflow?`Full state would exceed this rank’s slot by ${num(total-plan.slotBytes)} B.`:`Full state would fit, leaving ${num(plan.slotBytes-total)} B after the reserve.`,`small ${overflow?'orange-text':'green-text'}`);
    }
    s+=r(16,319,1008,55,rec.recompute?'orange-fill':'green-fill');
    s+=t(30,342,`max rank total = ${num(rec.test)} B ${rec.recompute?'>':'≤'} ${num(plan.slotBytes)} B → ${phase===0?'predicted ':''}need_recompute = ${rec.recompute?1:0}`,'mono bold');
    s+=t(30,364,rec.recompute?(totals[plan.rank]<=plan.slotBytes?'Both ranks save only x, even though the inspected rank’s full state would fit.':'The worst-rank test selects the replay branch on every rank.'):'Both ranks keep eligible state; their saved byte extents differ with received rows.','small');
    s+=t(16,400,'BACKWARD PATH SELECTED BY THE FORWARD FLAG','small bold');
    s+=card(16,413,272,85,rec.recompute?'Retained source x':'Retained expert state',rec.recompute?`BF16 [2,256] · ${num(plan.input)} B`:'x_gathered + h1 + h3','green-fill');
    s+=t(29,483,rec.recompute?'Saved activation slot':`${num(rec.full[plan.rank])} B on rank ${plan.rank}`,'small');
    s+=card(385,413,272,85,rec.recompute?'Replay forward intermediates':'Rebuild temporary h2',rec.recompute?'Gather + GEMMs + SwiGLU':'SwiGLU reads retained h1','amber-fill');
    s+=t(398,483,rec.recompute?'Device-predicated replay':'Expert GEMMs need no replay','small');
    s+=card(754,413,270,85,'Run gradient consumers','WGRAD + activation derivatives','blue-fill')+t(767,483,'Release after the last reader','small');
    s+=wire('M288,455 L385,455','green',true,false)+t(336,445,'read','small green-text','middle');
    s+=wire('M657,455 L754,455','blue',true,false)+t(705,445,'consume','small blue-text','middle');
    s+=t(16,522,backward?'The bars recall this layer’s forward test. Its saved flag selects the backward path; no new decision is made.':'The bars test a candidate allocation. Only the chosen branch is retained; its flag is saved for backward.','small muted');
    return finish(s);
  }
  function retention(layout,selected){start('retention');let s='',items=layout.items,slots=items.filter(i=>i.where==='slot'),scratch=items.filter(i=>i.where==='scratch');
    s+=t(16,24,`Physical slot ${layout.slot} · grows upward from offset ${hex(layout.start)}`,'bold')+t(1024,24,`need_recompute = ${layout.mode==='recompute'?1:0}`,'mono','end');
    s+=r(16,45,1008,95,'panel');let cursor=29;
    const slotW=(980-(slots.length-1)*9)/slots.length;
    for(const item of slots){s+=r(cursor,61,slotW,57,item.name===selected?'blue-fill':'green-fill')+t(cursor+9,84,item.name,'small')+t(cursor+9,108,bytes(item.bytes),'small');cursor+=slotW+9;}
    s+=t(16,163,`Region cards: ${bytes(layout.slotUsed)} occupied out of ${bytes(layout.slotBytes)}. Offsets specify byte extents; card widths are schematic.`,'small muted');
    s+=t(16,194,'SHARED SCRATCH · forward operands; not retained as independent saved state','small muted')+r(16,213,1008,83,'panel');
    const unique=scratch.filter((item,i,a)=>a.findIndex(v=>v.offset===item.offset)===i);let xx=29,columns=unique.length,w=(980-(columns-1)*9)/columns;
    unique.forEach(item=>{let title=item.name;if(layout.precision==='mxfp8'&&title.startsWith('x row'))title=title.includes('scales')?'x/h2 row scales':'x/h2 row qdata';s+=r(xx,230,w,49,item.name===selected?'blue-fill':'amber-fill')+t(xx+9,250,title,'small')+t(xx+9,270,hex(item.offset),'mono muted');xx+=w+9;});
    s+=r(16,324,1008,54,layout.mode==='save'?'green-fill':'orange-fill')+t(29,346,layout.mode==='save'?(layout.precision==='bf16'?'Backward reads saved x_gathered/h1/h3; SwiGLU rebuilds temporary h2.':'Backward reads saved column bundles and h1/h3; forward GEMMs need no replay.'):'Backward republishes saved x and reconstructs expert intermediates in scratch.','small bold')+t(29,367,'Each tensor’s consumer and exact offset are listed below. Region cards are schematic; the offset table gives the precise byte extents.','small muted');return finish(s);
  }
  function lifetimes(layout){
    start('lifetimes');const bf=layout.precision==='bf16',save=layout.mode==='save';let rows;
    const math={
      xg:[String.raw`x_{\mathrm g}=\operatorname{gather}(x)`,String.raw`\mathrm{grad}_{x_{\mathrm g}}=W_{13}^{\mathsf T}\mathrm{grad}_{h_1}`,String.raw`\mathrm{grad}_{W_{13}}\mathrel{+{=}}\mathrm{grad}_{h_1}x_{\mathrm g}^{\mathsf T}`],
      h1:[String.raw`h_1=W_{13}x_{\mathrm g}=[g;u]`,String.raw`\mathrm{grad}_{h_1}=[dg;du]`,'Backward needs the gate/up values'],
      h2:[String.raw`h_2=\operatorname{SiLU}(g)\odot u`,String.raw`\mathrm{grad}_{h_2}=W_2^{\mathsf T}\mathrm{grad}_{h_3}`,String.raw`\mathrm{grad}_{W_2}\mathrel{+{=}}\mathrm{grad}_{h_3}h_2^{\mathsf T}`],
      h3:[String.raw`h_3=W_2h_2`,String.raw`\mathrm{grad}_{h_3}=\mathrm{score}\,\mathrm{grad}_y`,String.raw`\mathrm{grad}_{\mathrm{score}}=\mathrm{grad}_y^{\mathsf T}h_3`],
      x:[String.raw`x=\text{caller input}`,String.raw`\mathrm{grad}_x=\sum_{\mathrm{routes}}\mathrm{grad}_{x_{\mathrm g}}`,'Replay gathers x into expert rows'],
      row:[String.raw`Q_{\mathrm{row}}(x_{\mathrm g}),\;Q_{\mathrm{row}}(h_2)`,'W13 and W2 forward operands','Row bytes can be reused after MMA']
    };
    if(save&&bf)rows=[['x_gathered',[[0,8,'green']],6,'xg'],['h1',[[0,8,'green']],5,'h1'],['h2 · scratch',[[1,3,'amber'],[3,5,'amber']],4,'h2'],['h3',[[2,8,'green']],3,'h3']];
    else if(save)rows=[['x column bundle',[[0,8,'green']],6,'xg'],['h1',[[0,8,'green']],5,'h1'],['h2 column bundle',[[1,8,'green']],4,'h2'],['h3',[[2,8,'green']],3,'h3'],['x/h2 row scratch',[[0,1,'amber'],[1,3,'amber']],2,'row']];
    else rows=[['Saved source x',[[0,8,'green']],3,'x'],[bf?'x_gathered':'x column bundle',[[0,3,'amber'],[3,7,'amber']],6,'xg'],['h1',[[0,3,'amber'],[3,6,'amber']],5,'h1'],[bf?'h2':'h2 column bundle',[[1,3,'amber'],[3,5,'amber']],4,'h2'],['h3 · combine HBM',[[2,3,'blue'],[3,4,'blue']],3,'h3']];
    const x0=560,unit=107.5;let s=t(16,20,'FORWARD PRODUCER / BACKWARD DEPENDENCE','small bold');
    const headers=[['W13','forward'],['SwiGLU','forward'],['W2','forward'],['Setup /','rebuild'],['W2','grad_h2'],['SwiGLU','grad_h1'],['W13','grad_xg'],['Release','slot']];
    headers.forEach((h,i)=>{s+=t(x0+i*unit+unit/2,20,h[0],'dependency-heading','middle')+t(x0+i*unit+unit/2,41,h[1],'dependency-subhead','middle');});
    s+=l(x0+3*unit-4,51,x0+3*unit-4,565,'line','stroke-dasharray="3 3"');
    rows.forEach((row,i)=>{
      const y=65+i*100;s+=t(16,y+15,row[0],'dependency-name');
      math[row[3]].forEach((tex,j)=>s+=j===0||tex.includes('=')?equation(j===0?260:16,y+[15,45,77][j],tex,j===0?'blue-text':'muted',26,j===0?290:532):t(16,y+(row[3]==='x'&&j===2?87:[15,45,77][j]),tex,'dependency-detail muted'));
      s+=l(x0,y+67,1420,y+67);
      for(const[a,b,color]of row[1])s+=r(x0+a*unit,y+13,(b-a)*unit-8,29,`${color}-fill`);
      const xx=x0+(row[2]+.5)*unit;s+=`<path d="M${xx},${y+7} l6,8 -6,8 -6,-8Z" fill="${C('purple')}"/>`;
    });
    s+=r(16,575,1408,74,'purple-fill')+equation(29,600,String.raw`dg=\mathrm{grad}_{h_2}\odot u\odot\operatorname{SiLU}'(g),\qquad du=\mathrm{grad}_{h_2}\odot\operatorname{SiLU}(g)`,'',26,1360)+equation(29,633,String.raw`\operatorname{SiLU}'(g)=\operatorname{sigmoid}(g)\odot\bigl(1+g\odot(1-\operatorname{sigmoid}(g))\bigr)`,'',26,1360);
    s+=equation(16,668,String.raw`x_{\mathrm g}=\texttt{x\_gathered}\;;\quad h_1=\texttt{h1}\;;\quad \mathrm{grad}_{h_1}=\texttt{grad\_h1}`,'muted',18,1360);
    return finish(s);
  }
  function kernel(p,bm,bn,tail,kSlice){start('kernel');const bk=64,valid=tail?Math.floor(bm*.703125):bm,scale=140/256,aw=bk*scale,ah=bm*scale,bw=bn*scale,bh=bk*scale,cw=bn*scale,ch=bm*scale;let s='';
    s+=r(16,10,189,247,'blue-fill')+t(30,35,'PEER SYMMETRIC HBM','small bold')+t(30,58,'Complete source rows','small');s+=grid(40,84,140,ah,16,16,'blue',valid/bm)+t(30,243,`${valid}/${bm} valid rows`,'small');
    s+=r(223,10,801,247,'green-fill','stroke-opacity=".45"')+t(239,35,'CURRENT GPU · LOCAL HBM','small green-text')+t(250,69,'Gathered M block','small')+grid(253,84,140,ah,16,16,'blue',valid/bm)+r(253+kSlice*aw,84,aw,ah,'','fill="none" stroke="var(--amber)" stroke-width="2"')+t(253,243,`[${valid}, 256] valid · BF16`,'mono');
    s+=t(452,69,'Weight B','small')+grid(452,84,bw,bh,4,16,'green')+t(452,155,`[${bk}, ${bn}] BF16`,'mono')+t(452,179,bytes(bk*bn*2),'small muted');
    s+=t(839,69,'h1 store window','small')+grid(839,84,cw,ch,16,16,'green',valid/bm)+t(839,242,`[${valid}, ${bn}] valid BF16`,'mono');
    s+=r(223,292,801,271,'panel')+t(239,317,'CTA / cluster tile pipeline','small orange-text');
    s+=r(236,337,309,211,p===2?'blue-fill':'box')+t(250,359,'TMA stages A and B in SMEM','small bold')+grid(253,380,aw,ah,16,4,'blue',valid/bm)+grid(393,380,bw,bh,4,16,'green')+t(393,459,`B: ${bk} × ${bn}`,'mono')+t(393,488,`A: ${bm} × ${bk}`,'mono');
    s+=r(562,337,230,211,p===3?'orange-fill':'box')+t(578,359,'MMA → TMEM C','small bold')+grid(602,380,cw,ch,16,16,'orange',valid/bm)+t(577,535,`FP32 · ${bytes(bm*bn*4)}`,'small');
    s+=r(813,337,211,211,p===4?'green-fill':'box')+t(828,359,'Epilogue C','small bold')+grid(850,380,cw,ch,16,16,'green',valid/bm)+t(828,535,'FP32 → BF16','small');
    s+=card(650,182,167,61,'tile_ready(M)','GPU counter / signal',p===1?'purple-fill':'box')+(p===1?wire('M733,243 L733,283 L550,283 L550,328 L535,337','purple',true,true,true,'Control: tile_ready(M) counter',{x:662,y:291,lines:['tile_ready(M)','GPU counter'],point:[733,283]}):'');
    s+=wire(`M180,${84+ah/2} L253,${84+ah/2}`,'blue',p===0,true,true,`source rows [${valid},256] · BF16`,{x:176,y:84+ah/2-76,lines:['x rows',`${valid}×256`,'BF16'],point:[216,84+ah/2]})+wire(`M393,${84+ah/2} L430,${84+ah/2} L430,327 L270,327 L270,337`,'blue',p===2,true,true,`A tile [${bm},64] · BF16`,{x:443,y:260,lines:[`A [${bm},64]`,'BF16 → SMEM'],point:[430,282]})+wire(`M${452+bw},101 L640,101 L640,327 L514,327 L514,337`,'green',p===2,true,true,`B tile [64,${bn}] · BF16`,{x:654,y:260,lines:[`B [64,${bn}]`,'BF16 → SMEM'],point:[640,282]})+wire('M545,452 L562,452','orange',p===3,true,true,`SMEM A [${bm},64] and B [64,${bn}]`,{x:495,y:260,lines:[`A [${bm},64]`,`B [64,${bn}]`,'BF16 operands'],point:[553,452]})+wire('M792,452 L813,452','orange',p===4,true,true,`C accumulator [${bm},${bn}] · FP32`,{x:747,y:260,lines:[`C [${bm},${bn}]`,'FP32'],point:[802,452]})+wire('M922,337 L922,257','green',p===5,true,true,`h1 store [${valid},${bn}] · BF16`,{x:801,y:261,lines:[`h1 [${valid},${bn}]`,'BF16 → HBM'],point:[922,283]});
    s+=t(216,84+ah/2+(p===0?22:-18),'NVLink','tiny blue-text','middle')+t(16,575,p<=1?'Gather completes full rows → TMA selects the highlighted 64-column slice.':'Full-row gather and K-slice staging are different operations; barriers order their handoff.','small purple-text');
    s+=r(16,582,1008,34,'box')+t(29,605,`K slice ${kSlice+1} of 4: columns ${kSlice*bk}–${(kSlice+1)*bk-1}; C accumulates across K slices. Full C tile = ${bm} × ${bn}, valid rows = ${valid}.`,'small');return finish(s);
  }
  function precision(format,outlier,selected){start('precision');const q=Moe.quantBlock(outlier);let s=t(16,23,format==='nvfp4'?'Two 16-value blocks from one BF16 source row':format==='bf16'?'32 BF16 source values · two data bytes per value':'BF16 values (blue) and reconstructed scale × q (green)','bold');
    s+=r(901,40,123,147,'box')+t(914,62,'Value range','small muted')+t(914,90,`+${q.amax}`,'mono')+t(914,120,'0','mono muted')+t(914,167,`−${q.amax}`,'mono');const baseline=113,span=70,plotW=864,cell=plotW/32;s+=l(16,baseline,880,baseline,'faint');
    q.x.forEach((v,i)=>{const xx=18+i*cell,h=Math.abs(v)/q.amax*span,yy=v>=0?baseline-h:baseline;s+=`<rect x="${xx}" y="${yy}" width="9" height="${Math.max(.3,h)}" fill="${C('blue')}"/>`;if(format==='mxfp8'){const d=q.decoded[i],dh=Math.abs(d)/q.amax*span;s+=`<rect x="${xx+11}" y="${d>=0?baseline-dh:baseline}" width="9" height="${Math.max(.3,dh)}" fill="${C('green')}"/>`;}if(i===selected)s+=r(xx-2,40,24,148,'','fill="none" stroke="var(--amber)"');});
    if(format==='bf16'){
      s+=card(16,211,864,63,'BF16 data stays BF16','No amax / scale / qdata producer is required for the BF16 expert operands.','blue-fill');s+=wire('M440,187 L440,211','blue',true,false)+wire('M440,274 L440,303','blue',true,false);s+=t(16,294,'64 BF16 data bytes','small muted');
      q.x.forEach((v,i)=>{const x=16+(i%16)*54,y=303+Math.floor(i/16)*51;s+=r(x,y,46,39,'blue-fill')+l(x+23,y+4,x+23,y+34,'blue')+t(x+23,y+25,'2 B','small','middle');});
      s+=card(901,303,123,90,'No scales','BF16 data','box')+r(16,415,1008,40,'box')+t(29,441,'Expert products accumulate in FP32 TMEM; source and stored expert activations remain BF16.','small');
    }else{
      const nv=format==='nvfp4';s+=card(16,211,240,63,'1 · Find magnitudes',nv?'FP32 row / block reductions':`amax(|x|) = ${q.amax}`,'blue-fill')+card(278,211,240,63,'2 · Select scales',nv?'block E4M3 + global factors':`E8M0 power-of-two s = ${q.scale}`,'purple-fill')+card(540,211,340,63,'3 · Round and encode',nv?'E2M1 values → two values / byte':'q = nearest-even E4M3(x / s)','green-fill');
      s+=wire('M130,188 L130,211','blue',true,false)+wire('M710,188 L710,211','blue',true,false)+wire('M256,243 L278,243','purple',true,false)+wire('M518,243 L540,243','purple',true,false)+wire('M710,274 L710,303','green',true,false)+wire('M405,274 L405,283 L962,283 L962,303','purple',true,false);
      s+=t(16,294,nv?'16 packed qdata bytes · labels identify logical values':'32 E4M3 qdata bytes · select a byte to inspect it','small muted');
      if(!nv){q.bytes.forEach((b,i)=>{const x=16+(i%16)*54,y=303+Math.floor(i/16)*51;s+=`<g class="clickable" data-byte="${i}" tabindex="0" role="button" aria-label="Inspect value ${i}, qdata byte ${hex(b)}">`+r(x,y,46,39,i===selected?'amber-fill':'green-fill')+t(x+23,y+25,b.toString(16).toUpperCase().padStart(2,'0'),'mono','middle')+'</g>';});s+=card(901,303,123,90,'E8M0 scale',hex(q.scaleByte),'purple-fill')+t(914,378,`s = ${q.scale}`,'small');s+=r(16,415,1008,40,'box')+t(29,441,`s = 2^ceil(log₂(amax / 448)) = ${q.scale}; decode q × s to inspect the rounding error above.`,'small');}
      else{for(let i=0;i<16;i++){const x=16+i*54;s+=r(x,303,46,46,'orange-fill')+l(x+23,308,x+23,344,'orange')+t(x+11,332,`${i*2}`,'small','middle')+t(x+34,332,`${i*2+1}`,'small','middle');}s+=t(901,326,'NVFP4','bold orange-text')+t(901,351,'Inference only','small')+card(16,384,310,71,'2 block-scale bytes','E4M3 · one per 16 values','purple-fill')+card(351,384,323,71,'FP32 inverse global scale','One per complete source row','amber-fill')+card(700,384,324,71,'Row stride aligned to 16 bytes','qdata + block scales + row inverse','blue-fill');}
    }
    return {svg:finish(s),q};
  }
  function orientation(){let a='',b='';for(let row=0;row<8;row++)for(let col=0;col<8;col++){a+=r(20+col*22,20+row*22,20,20,col<4?'blue-fill':'green-fill');b+=r(20+col*22,20+row*22,20,20,row<4?'purple-fill':'amber-fill');}a+=t(210,47,'← 32 columns','small')+t(210,80,'per logical row','small')+t(20,226,'8 × 8 display bins for a 64 × 64 operand','tiny muted');b+=t(210,47,'32 rows ↓','small')+t(210,80,'per logical column','small')+t(20,226,'Each bin represents 8 × 8 logical values.','tiny muted');return {a,b};}
  function fusion(p,pipeline,focus=0){start('fusion');const mega=pipeline==='mega',step=c=>({g:c,w13:c+1,sw:c+(mega?2:4),w2:c+(mega?3:5),scatter:c+(mega?4:6)}),st=step(focus);let s='<g class="fusion-flow">';
    if(mega)s+=r(8,8,1024,303,'green-fill','stroke-dasharray="5 4" stroke-opacity=".55"');else s+=r(8,8,1024,124,'blue-fill','stroke-dasharray="5 4" stroke-opacity=".55"')+r(8,173,1024,139,'orange-fill','stroke-dasharray="5 4" stroke-opacity=".55"');
    s+=t(20,28,mega?'ONE KERNEL GRID · SHARED WARP ROLES':'LAUNCH 1 · dispatch + W13','small bold');s+=t(1018,28,`FOLLOW CHUNK ${focus}`,'small amber-text','end');if(!mega)s+=t(1018,195,'LAUNCH 2 · SwiGLU + W2 + combine','small bold','end');
    const nodes=[{x:20,y:45,n:'Gather / quant roles',sub:'publishes x qdata/scales',at:st.g,col:'blue'},{x:278,y:45,n:'x row bundle',sub:p>st.g?'ready for W13':'waiting for publication',at:st.w13,col:'green'},{x:536,y:45,n:'TMA / MMA: W13',sub:'SMEM → FP32 TMEM',at:st.w13,col:'orange'},{x:794,y:45,n:'Epilogue: W13',sub:p>st.w13?'h1 BF16 + ready signal':'wait for W13',at:st.sw,col:'green'},{x:20,y:213,n:'SwiGLU / quant roles',sub:'reads h1; publishes h2',at:st.sw,col:'purple'},{x:278,y:213,n:'h2 row bundle',sub:p>st.sw?'qdata + scales ready':'wait for quantization',at:st.w2,col:'green'},{x:536,y:213,n:'TMA / MMA: W2',sub:'SMEM → FP32 TMEM',at:st.w2,col:'orange'},{x:794,y:213,n:'W2 scatter',sub:'BF16 → combine HBM',at:st.scatter,col:'green'}];
    nodes.forEach(n=>s+=`<g class="flow-card">`+card(n.x,n.y,226,67,n.n,n.sub,p===n.at?`${n.col}-fill`:'box')+'</g>');
    s+=wire('M246,80 L278,80','blue',p===st.w13,true,true,'x tile: E4M3 qdata + E8M0 scales',{x:211,y:119,lines:['x qdata/scales','MXFP8 · [512,D]'],point:[262,80]})+wire('M504,80 L536,80','orange',p===st.w13,true,true,'x / weight tiles from SMEM → MMA',{x:451,y:119,lines:['x + weight tiles','SMEM → MMA'],point:[520,80]})+wire('M762,80 L794,80','green',p===st.sw,true,true,'C accumulator tile · FP32',{x:702,y:119,lines:['C [B_M,B_N]','FP32'],point:[778,80]});
    s+=wire('M908,114 L908,166 C908,172 132,166 132,211','purple',p===st.sw,true,true,'completed h1 values [2F] · BF16',{x:69,y:119,lines:['h1 [2F] · BF16','→ SwiGLU'],point:[207,182]})+wire('M246,246 L278,246','blue',p===st.w2,true,true,'h2: E4M3 qdata + E8M0 scales',{x:211,y:284,lines:['h2 qdata/scales','MXFP8 · [512,F]'],point:[262,246]})+wire('M504,246 L536,246','orange',p===st.w2,true,true,'h2 / weight tiles from SMEM → MMA',{x:451,y:284,lines:['h2 + weight tiles','SMEM → MMA'],point:[520,246]})+wire('M762,246 L794,246','blue',p===st.scatter,true,true,'C accumulator tile · FP32',{x:702,y:284,lines:['C [B_M,B_N]','FP32'],point:[778,246]});
    s+='</g><g class="fusion-chunks">';
    s+=t(20,353,'CHUNK STATE · filled = completed, bright = working, muted = waiting','small muted');
    const labels=['Gather / quant','W13','SwiGLU / quant','W2','Scatter'];labels.forEach((label,i)=>s+=t(200+i*164,373,label,'small bold'));
    for(let c=0;c<3;c++){const y=390+c*45,st=step(c);s+=t(20,y+25,`chunk ${c}${c===focus?' ←':''}`,c===focus?'mono amber-text':'mono');Object.values(st).forEach((at,i)=>{const state=p>at?'done':p===at?'working':'waiting',color=['blue','orange','purple','orange','green'][i];s+=r(200+i*164,y,150,34,p>=at?`${color}-fill`:'box',p>at?'opacity=".65"':'')+t(213+i*164,y+23,state,'small');});}
    s+=t(20,530,'Ready counters order reads; consumption credits protect stage reuse. Storage remains context-owned and allocated.','small purple-text');
    s+='</g><g class="fusion-hardware">';
    s+=r(1060,8,522,534,'panel')+t(1080,42,'HARDWARE: ONE RESIDENT CTA','bold');
    if(mega){
      s+=t(1080,76,'One SM hosts this complete role set.','small')+t(1080,105,'Narrow MXFP8 · 16 warps × 32 = 512 threads','small muted');
      const roles=[['Warps 0–3 · gather / quant','Peer rows → x qdata/scales','blue'],['Warps 4/5/6 · TMA-A / MMA / TMA-B','Warp 7: no mainloop role','orange'],['Warps 8–11 · epilogue','W13 h1 stores; W2 peer scatter','green'],['Warps 12–15 · SwiGLU / quant','h1 → h2 qdata/scales','purple']];
      roles.forEach(([label,detail,color],i)=>s+=r(1080,129+i*81,482,70,`${color}-fill`)+t(1094,154+i*81,label,'small bold')+t(1094,182+i*81,detail,'small'));
      s+=t(1080,472,'The same TMA/MMA and epilogue roles','small bold')+t(1080,497,'serve both W13 and W2 work records.','small bold')+t(1080,525,'Waiting warps stay resident; ready warps issue.','small muted');
    }else{
      s+=t(1080,83,'Fixed roles coexist within each CTA.','small')+card(1080,119,482,110,'Launch 1 CTA','gather/quant + TMA + MMA + epilogue','blue-fill')+t(1094,209,'SM schedulers issue eligible warps.','small');
      s+=wire('M1321,229 L1321,293','purple',true,false)+t(1350,262,'launch completion','small purple-text');
      s+=card(1080,293,482,110,'Launch 2 CTA','SwiGLU/quant + TMA + MMA + scatter','orange-fill')+t(1094,383,'A new CTA lifetime starts for launch 2.','small');
      s+=t(1080,460,'Staged uses its own configured warp split.','small muted')+t(1080,500,'Scratch can be reused after launch 1 finishes.','small');
    }
    s+='</g>';return finish(s);
  }
  const slotColors=['blue','green','orange','purple','pink','cyan'];
  function liveness(p){
    start('liveness');const stage={...Moe.lifetimes('stage'),...Moe.intervals('stage')},microbatch={...Moe.lifetimes('microbatch'),...Moe.intervals('microbatch')},action=Moe.schedule[p],stageId=Number(action[0]),kind=action[1],mb=Number(action.slice(2));
    const stageBudget=stage.slots*stage.depth*stage.layerBytes,microbatchBudget=microbatch.slots*microbatch.depth*microbatch.layerBytes,saving=microbatchBudget-stageBudget,percent=100*saving/microbatchBudget;
    const comparison=`${stage.slots} stage–microbatch slots × ${bytes(stage.depth*stage.layerBytes)} = ${bytes(stageBudget)}; ${microbatch.slots} whole-microbatch slots × ${bytes(microbatch.depth*microbatch.layerBytes)} = ${bytes(microbatchBudget)}. Smaller stage slots save ${bytes(saving)} (${num(percent)}%) because they hold only one stage’s ${stage.depth} MoE layers, rather than all ${microbatch.depth} MoE layers on this rank.`;
    const pairRemainder=stage.list.filter(iv=>iv.mb===mb&&iv.start<=p&&iv.end>p+1),whole=microbatch.list.find(iv=>iv.mb===mb),stillLive=whole.start<=p&&whole.end>p+1;
    const summaries={
      stage:[`${action} ${kind==='F'?'starts':'ends'} (stage ${stageId}, microbatch ${mb}).`,stage.counts[p]===0?'No stage–microbatch lifetimes remain live.':kind==='B'&&pairRemainder.length?`Stage ${pairRemainder[0].stage} / MB ${mb} is still a live pair.`:stage.counts[p]===stage.peak?`${stage.peak} pairs overlap: the whole-trace maximum.`:`${stage.counts[p]} stage–microbatch pairs now overlap.`],
      microbatch:[kind==='F'?(whole.start===p?`${action} starts microbatch ${mb}’s local lifetime.`:`${action} adds stage ${stageId} to live microbatch ${mb}.`):stillLive?`Microbatch ${mb} remains live after ${action}.`:`${action} ends microbatch ${mb}’s local lifetime.`,kind==='B'&&stillLive?'Its other local stage still needs backward.':microbatch.counts[p]===0?'No whole-microbatch lifetimes remain live.':microbatch.counts[p]===microbatch.peak?`${microbatch.peak} microbatches overlap: the whole-trace maximum.`:`${microbatch.counts[p]} whole microbatches now overlap.`]
    };
    const takeaway=stage.counts[p]===0&&microbatch.counts[p]===0?'The last terminal backward closes both kinds of lifetime. All local saved-state lifetimes have ended.':stage.counts[p]===stage.peak&&microbatch.counts[p]===microbatch.peak?comparison:kind==='B'&&stillLive?'A stage’s backward can end its pair’s lifetime while the whole microbatch still has another local stage to finish.':kind==='B'?'This backward completes the microbatch’s final local stage, ending both that stage’s pair and the whole-microbatch interval.':whole.start===p?'A new microbatch opens a whole-microbatch interval and its first stage–microbatch interval.':'Starting another local stage adds a stage–microbatch interval while the existing whole-microbatch lifetime continues.';
    let s=t(16,24,`AFTER ${action} · ACTION ${p} COMPLETED`,'bold')+t(1024,24,'Maxima use the complete 32-action trace','small muted','end');
    const sharedPeak=Math.max(stage.peak,microbatch.peak);
    for(const [mode,info,x]of [['stage',stage,16],['microbatch',microbatch,532]]){
      const startX=x+68,step=408/32;s+=r(x,40,492,533,'panel');
      s+=t(x+14,63,mode==='stage'?'Stage–microbatch lifetimes':'Whole-microbatch lifetimes','bold');
      s+=t(x+14,87,`Live: ${info.counts[p]}`,'large bold')+t(x+278,87,`Max live: ${info.peak}`,'large bold');
      s+=t(x+14,116,'Slots','small muted');
      for(let slot=0;slot<info.slots;slot++){const xx=x+70+slot*65,color=slotColors[slot];s+=r(xx,105,11,11,`${color}-fill`)+t(xx+17,116,`S${slot}`,'small');}
      for(let i=0;i<=32;i+=4)s+=t(startX+i*step,136,String(i),'tiny muted','middle');
      for(let b=0;b<8;b++){
        const y=151+b*31;s+=t(x+14,y+18,`MB ${b}`,'small bold')+l(startX,y+28,startX+408,y+28);
        for(const iv of info.list.filter(iv=>iv.mb===b)){
          const live=iv.start<=p&&iv.end>p+1,color=slotColors[iv.slot],yy=mode==='microbatch'?y+6:y+(iv.stage===0?2:15),h=mode==='microbatch'?17:10;
          if(mode==='stage')s+=t(startX-6,yy+9,String(iv.stage),'tiny muted','end');
          s+=r(startX+iv.start*step,yy,(iv.end-iv.start)*step-1,h,`${color}-fill`,`data-lifetime-policy="${mode}" data-slot="${iv.slot}" data-owner="${iv.key}" ${live?'stroke-width="2"':iv.start>p?'opacity=".2"':'opacity=".4"'}`);
        }
      }
      const pointer=startX+(p+1)*step;s+=l(pointer,142,pointer,402,'amber','stroke-dasharray="4 3"');
      s+=t(x+14,418,'LIVE AFTER EACH ACTION','small bold');const baseline=487,scale=40/sharedPeak,peakY=baseline-info.peak*scale;
      s+=l(startX,peakY,startX+408,peakY,'line','stroke-dasharray="3 4"')+t(x+14,peakY+4,`${info.peak} max`,'tiny muted');
      info.counts.forEach((n,i)=>{const live=info.list.filter(iv=>iv.start<=i&&iv.end>i+1).sort((a,b)=>a.slot-b.slot);live.forEach((iv,j)=>s+=`<rect x="${startX+i*step}" y="${baseline-(j+1)*scale}" width="${step-1}" height="${scale}" fill="${C(slotColors[iv.slot])}" opacity="${i>p?.25:.85}"/>`);});
      s+=l(pointer,438,pointer,493,'amber','stroke-dasharray="4 3"')+t(startX+(p+.5)*step,baseline-info.counts[p]*scale-5,String(info.counts[p]),'small bold','middle');
      s+=t(x+14,513,`Budget: ${info.slots} slots × ${info.depth} layers × ${bytes(info.layerBytes)} = ${bytes(info.slots*info.depth*info.layerBytes)}`,'small bold');
      summaries[mode].forEach((line,i)=>s+=t(x+14,537+i*22,line,'small'));
    }
    s+=t(16,602,'Color = slot, reused after release. Left rows label stages 0 / 2; right rows span all local stages.','small muted');
    return {svg:finish(s),stage,microbatch,summaries,takeaway,comparison};
  }
  function slots(policy,p){start('slots');const info=Moe.intervals(policy),x0=129,step=60;let s='',sticky='';
    Moe.schedule.forEach((a,i)=>{const x=x0+i*step;s+=t(x+25,24,String(i),'small muted','middle')+r(x,37,52,32,i===p?'amber-fill':'box',i>p?'opacity=".4"':'')+t(x+26,59,a,'mono','middle');});
    s+=t(x0,93,'Outline = acquire · dashed line = completed action','small muted');
    sticky+=t(15,93,'COLOR = SLOT','small muted');
    for(let slot=0;slot<info.slots;slot++){const y=104+slot*47,color=slotColors[slot];sticky+=r(0,y-5,122,48,'panel','stroke="none"')+r(12,y+12,11,11,`${color}-fill`)+t(31,y+24,`SLOT ${slot}`,'mono');s+=l(x0,y+36,2050,y+36);info.list.filter(iv=>iv.slot===slot).forEach(iv=>{const x=x0+iv.start*step,w=(iv.end-iv.start)*step-7,live=iv.start<=p&&iv.end>p+1,acquire=iv.start===p;const label=policy==='stage'?`stage ${iv.stage} / mb ${iv.mb}`:`mb ${iv.mb} · all local stages`;s+=r(x,y,w,34,`${color}-fill${acquire?' slot-acquired':''}`,`data-slot="${slot}" data-owner="${iv.key}" ${live||acquire?'':iv.start>p?'opacity=".2"':'opacity=".45"'}`)+t(x+10,y+23,label,'small','start','data-owner-label');});}
    const counts=Moe.lifetimes(policy).counts;
    s+=l(x0+(p+1)*step-5,76,x0+(p+1)*step-5,104+info.slots*47,'amber','stroke-dasharray="4 4"')+`<g data-sticky-slot>${r(0,0,122,78,'panel','stroke="none"')}${t(15,31,'ACTION','small muted')}${sticky}</g>`;return {svg:finish(s),info,counts,height:124+info.slots*47};
  }
  return {esc,mathHTML,num,hex,bytes,t,r,l,card,grid,badge,moe,overview,tokens,tokenPhases,consumption,calculations,packing,lookahead,retention,lifetimes,renderMath,positionMath,kernel,precision,orientation,fusion,liveness,slots};
})();
