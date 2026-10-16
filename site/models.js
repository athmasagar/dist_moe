'use strict';
// Arithmetic follows the public ModelConfig and BlockscaledStorageConfig methods.
// The browser model covers ordinary BF16 and staged/Mega MXFP8 training only.
window.Moe = (() => {
  const align = (n, multiple = 128) => Math.ceil(n / multiple) * multiple;
  const sum = (parts) => parts.reduce((s, p) => s + p.bytes, 0);
  const part = (name, bytes, formula, dtype = 'BF16') => ({ name, bytes: align(bytes), formula, dtype });
  const row = (r, d) => ({ q: align(r * d), s: align(r * 4 * Math.ceil(d / 32)), bytes: align(r * d) + align(r * 4 * Math.ceil(d / 32)) });
  const col = (r, d) => ({ q: align(r * d), s: align(d * Math.ceil(r / 32)), bytes: align(r * d) + align(d * Math.ceil(r / 32)) });
  function capacity(c, factor) {
    const raw = Math.min(Math.floor(c.T * c.K * factor + .5), c.T * c.EP * Math.min(c.K, c.E / c.EP));
    return c.precision === 'bf16' ? raw : Math.floor((raw + c.E / c.EP * 127) / 128) * 128;
  }
  function savedParts(c, r) {
    if (c.precision === 'bf16') return [part('x_gathered', r * c.D * 2, 'align128(R × D × 2)'), part('h1', r * 2 * c.F * 2, 'align128(R × 2F × 2)'), part('h3', c.T * c.K * c.D * 2, 'align128(T × K × D × 2)')];
    return [part('x column qdata', col(r,c.D).q, 'align128(R × D)', 'E4M3'),part('x column scales',col(r,c.D).s,'align128(D × ceil(R / 32))','E8M0'),part('h1',r*2*c.F*2,'align128(R × 2F × 2)'),part('h2 column qdata',col(r,c.F).q,'align128(R × F)','E4M3'),part('h2 column scales',col(r,c.F).s,'align128(F × ceil(R / 32))','E8M0'),part('h3',c.T*c.K*c.D*2,'align128(T × K × D × 2)')];
  }
  function scratchParts(c, r) {
    if (c.precision === 'bf16') {
      const p=[part('Replayed x_gathered',r*c.D*2,'align128(R × D × 2)'),part('Replayed h1',r*2*c.F*2,'align128(R × 2F × 2)'),part('grad_h1 / h2 reuse',r*2*c.F*2,'align128(R × 2F × 2)'),part('grad_h2',r*c.F*2,'align128(R × F × 2)')];
      if(c.D>c.F)p.push(part('grad_h3 gathered',r*c.D*2,'align128(R × D × 2)'));
      return p;
    }
    const qx=row(r,c.D).bytes,qh=row(r,c.F).bytes,qg=row(r,2*c.F).bytes;
    const rowPeak=c.pipeline==='mega'?Math.max(qx+qh,qg):Math.max(qx,qh,qg);
    return [part('Shared row-operand scratch',rowPeak,c.pipeline==='mega'?'max(Qrow(R,D) + Qrow(R,F), Qrow(R,2F))':'max(Qrow(R,D), Qrow(R,F), Qrow(R,2F))','E4M3 + E8M0'),part('Replayed x column bundle',col(r,c.D).bytes,'Qcol(R,D)','E4M3 + E8M0'),part('Replayed h1',r*2*c.F*2,'align128(R × 2F × 2)'),part('Replayed h2 column bundle',col(r,c.F).bytes,'Qcol(R,F)','E4M3 + E8M0'),part('grad_h3 column bundle',col(r,c.D).bytes,'Qcol(R,D)','E4M3 + E8M0'),part('grad_h2',r*c.F*2,'align128(R × F × 2)'),part('grad_h1 column bundle',col(r,2*c.F).bytes,'Qcol(R,2F)','E4M3 + E8M0')];
  }
  function memoryBounds(c) {
    const deviceFactor=Math.min(c.deviceFactor,c.EP),totalFactor=c.vmm?Math.min(c.totalFactor,c.EP):deviceFactor;
    if(!Number.isFinite(deviceFactor)||deviceFactor<=0)throw new Error('Device scratch factor must be positive and finite.');
    if(!Number.isFinite(totalFactor)||totalFactor<deviceFactor)throw new Error('Total scratch factor must cover the device scratch factor.');
    const balanced=c.T*c.K,rb=capacity(c,1),rd=capacity(c,deviceFactor),rt=capacity(c,totalFactor);
    const minimum=align(c.L*align(c.T*c.D*2)),full=align(c.L*sum(savedParts(c,rb))),maximumUseful=align(c.L*sum(savedParts(c,rt)));
    return {balanced,rb,rd,rt,minimum,full,maximumUseful,optional:full-minimum,deviceFactor,totalFactor};
  }
  function slotFromFactor(factor,b) {
    if(!Number.isFinite(factor)||factor<0)throw new Error('Activation slot factor must be nonnegative and finite.');
    const selected=b.minimum+Math.ceil(factor*b.optional);
    if(!Number.isSafeInteger(selected)||selected>Number.MAX_SAFE_INTEGER-128)throw new Error('Slot request exceeds this calculator’s exact integer range.');
    return align(selected);
  }
  function factorForSlot(bytes,b) {
    if(bytes<=b.minimum||!b.optional)return 0;
    let factor=(align(bytes)-b.minimum)/b.optional;
    for(let i=0;i<3&&slotFromFactor(factor,b)>align(bytes);i++)factor-=Math.max(1,Math.abs(factor))*Number.EPSILON;
    return factor;
  }
  function memory(c) {
    const b=memoryBounds(c);
    if(c.activation_slot_bytes!=null&&c.activation_slot_capacity_factor!=null)throw new Error('Specify exact slot bytes or a slot factor, not both.');
    let slot=b.minimum;
    if(c.activation_slot_bytes!=null){
      if(!Number.isSafeInteger(c.activation_slot_bytes)||c.activation_slot_bytes<b.minimum||c.activation_slot_bytes>Number.MAX_SAFE_INTEGER-128)throw new Error('Exact slot bytes must be an integer at or above the minimum.');
      slot=align(c.activation_slot_bytes);
    }else if(c.activation_slot_capacity_factor!=null)slot=slotFromFactor(c.activation_slot_capacity_factor,b);
    const deviceParts=scratchParts(c,b.rd),totalParts=scratchParts(c,b.rt),deviceScratch=sum(deviceParts),totalScratch=sum(totalParts),hostScratch=Math.max(0,totalScratch-deviceScratch);
    return {...b,slot,deviceParts,totalParts,deviceScratch,totalScratch,hostScratch,activation:c.N*slot,device:slot*c.N+deviceScratch,virtual:slot*c.N+totalScratch,savedParts:savedParts(c,b.rb)};
  }
  // Four ordinary BF16 layers, exactly matching the device look-ahead branch.
  // Each layer routes 8 assignments across two ranks; no physical row padding.
  function packing(slotBytes=28672,rank=1) {
    const c={T:2,D:256,F:256,K:2,E:4,EP:2,precision:'bf16',pipeline:'staged'};
    const counts=[[4,4],[6,2],[3,5],[5,3]],depth=counts.length,input=align(c.T*c.D*2);
    const minimum=depth*input,maximum=Math.max(...[0,1].map(r=>counts.reduce((n,rows)=>n+sum(savedParts(c,rows[r])),0)));
    slotBytes=align(Math.max(minimum,Math.min(maximum,slotBytes)));
    const scratch=sum(scratchParts(c,6)),scratchEnd=2*slotBytes+scratch;
    // Both examples use physical slot 1. rank selects an EP rank, not a slot.
    const start=slotBytes,totals=[0,0],records=[];
    let cursor=start;
    for(let layer=0;layer<depth;layer++) {
      const rows=counts[layer],full=rows.map(r=>sum(savedParts(c,r))),before=[...totals];
      const reserve=(depth-layer-1)*input,candidate=before.map((n,r)=>n+full[r]);
      const worst=Math.max(...candidate),recompute=worst+reserve>slotBytes;
      const saved=recompute?[part('x',input,'align128(T × D × 2)')]:savedParts(c,rows[rank]);
      const base=cursor,items=saved.map(p=>{const item={...p,offset:cursor};cursor+=p.bytes;return item;});
      const scratchPartsNow=recompute?[part('x_gathered',rows[rank]*c.D*2,''),part('h1',rows[rank]*2*c.F*2,''),part('h2',rows[rank]*c.F*2,'')]:[part('h2',rows[rank]*c.F*2,'')];
      let high=scratchEnd;const forwardScratch=[...scratchPartsNow].reverse().map(p=>{high-=p.bytes;return {...p,offset:high};}).reverse();
      for(let r=0;r<2;r++)totals[r]+=recompute?input:full[r];
      const xgSize=align(rows[rank]*c.D*2),h1Size=align(rows[rank]*c.F*4),h2Size=align(rows[rank]*c.F*2);
      const gradH1=scratchEnd-h1Size,gradH2=gradH1-h2Size,bwdH1=recompute?gradH2-h1Size:items.find(i=>i.name==='h1').offset,bwdXg=recompute?bwdH1-xgSize:items.find(i=>i.name==='x_gathered').offset;
      const bwdOffsets=[bwdXg,bwdH1,scratchEnd-h2Size,gradH2,scratchEnd-h2Size-xgSize,gradH1];
      const backwardScratch=[...(recompute?[{name:'Replayed x_gathered',bytes:xgSize,offset:bwdXg},{name:'Replayed h1',bytes:h1Size,offset:bwdH1}]:[]),{name:'grad_h2',bytes:h2Size,offset:gradH2},{name:'h2 + grad_h3_gathered → grad_h1',bytes:h1Size,offset:gradH1}];
      records.push({layer,rows,full,before,reserve,worst,test:worst+reserve,recompute,base,end:cursor,items,forwardScratch,backwardScratch,bwdOffsets,after:[...totals]});
    }
    const snapshots=[{phase:0,kind:'empty',live:[],cursor:start,totals:[0,0],record:null}];
    for(let layer=0;layer<depth;layer++) snapshots.push({phase:layer+1,kind:'forward',live:records.slice(0,layer+1),cursor:records[layer].end,totals:records[layer].after,record:records[layer]});
    for(let layer=depth-1;layer>=0;layer--)snapshots.push({phase:2*depth-layer,kind:'backward',live:records.slice(0,layer),cursor:records[layer].base,totals:records[layer].before,record:records[layer]});
    return {c,counts,depth,input,minimum,maximum,slotBytes,scratch,scratchEnd,start,rank,records,snapshots};
  }
  const tokenData={A:{rank:0,index:0,experts:[0,2],scores:[.6,.4],outputs:[[2,4],[6,2]]},B:{rank:0,index:1,experts:[3,1],scores:[.25,.75],outputs:[[1,3],[5,7]]},C:{rank:1,index:0,experts:[2,0],scores:[.7,.3],outputs:[[4,2],[2,6]]},D:{rank:1,index:1,experts:[1,3],scores:[.5,.5],outputs:[[8,2],[0,6]]}};
  /** Resolve the documented BF16 routes in expert-major pointer order. */
  function routingRows(rank) {
    const rows=[];
    for(let expert=rank*2;expert<rank*2+2;expert++)for(const [name,value]of Object.entries(tokenData)){
      const k=value.experts.indexOf(expert);
      if(k>=0)rows.push({name,expert,k,owner:value.rank,token:value.index,gatherByte:value.index*512,returnByte:(value.index*2+k)*512,row:rows.length});
    }
    return rows;
  }
  function layout(precision,mode,slot=1) {
    const c={T:2,D:256,F:256,K:2,E:4,EP:2,precision,pipeline:'staged'},r=precision==='bf16'?4:256;
    const slotBytes=mode==='save'?sum(savedParts(c,r)):align(c.T*c.D*2),scratchCapacity=sum(scratchParts(c,r)),end=3*slotBytes+scratchCapacity,start=slot*slotBytes,items=[];let cursor=start,scratch=end;
    const add=(name,bytes,dtype,shape,where,why)=>{bytes=align(bytes);const offset=where==='slot'?cursor:scratch-bytes;items.push({name,bytes,dtype,shape,where,offset,why});if(where==='slot')cursor+=bytes;else scratch-=bytes;};
    if(mode==='recompute')add('x',c.T*c.D*2,'BF16','[2, 256]','slot','Replay the source input in backward');
    if(precision==='bf16'){
      if(mode==='save'){add('x_gathered',r*c.D*2,'BF16','[4, 256]','slot','W13 weight gradient');add('h1',r*2*c.F*2,'BF16','[4, 512]','slot','SwiGLU derivative and h2 replay');add('h3',c.T*c.K*c.D*2,'BF16','[2, 2, 256]','slot','Router-score gradient');add('h2',r*c.F*2,'BF16','[4, 256]','scratch','Temporary; rebuilt with SwiGLU');}
      else{add('h2',r*c.F*2,'BF16','[4, 256]','scratch','Replayed SwiGLU output');add('h1',r*2*c.F*2,'BF16','[4, 512]','scratch','Replayed W13 output');add('x_gathered',r*c.D*2,'BF16','[4, 256]','scratch','Replayed peer gather');}
    }else{
      const qr=row(r,c.D),qc=col(r,c.D),hf=col(r,c.F);const rowStart=end-Math.max(qr.bytes,row(r,c.F).bytes);
      const push=(name,bytes,offset,dtype,shape,where,why)=>items.push({name,bytes,dtype,shape,where,offset,why});
      push('x row qdata',qr.q,rowStart,'E4M3','[256, 256]','scratch','W13 FPROP operand');push('x row scales',qr.s,rowStart+qr.q,'E8M0','128-row blocked storage','scratch','W13 FPROP scales');
      push('h2 row qdata',qr.q,rowStart,'E4M3','[256, 256]','scratch','W2 FPROP; aliases x row storage');push('h2 row scales',qr.s,rowStart+qr.q,'E8M0','128-row blocked storage','scratch','W2 FPROP scales');scratch=rowStart;
      if(mode==='save'){add('x column qdata',qc.q,'E4M3','[256, 256]','slot','W13 WGRAD operand');add('x column scales',qc.s,'E8M0','[256, 8] logical transpose scales','slot','W13 WGRAD scales');add('h1',r*2*c.F*2,'BF16','[256, 512]','slot','SwiGLU backward');add('h2 column qdata',hf.q,'E4M3','[256, 256]','slot','W2 WGRAD operand');add('h2 column scales',hf.s,'E8M0','[256, 8] logical transpose scales','slot','W2 WGRAD scales');add('h3',2*2*256*2,'BF16','[2, 2, 256]','slot','Router-score gradient');}
      else{scratch-=qc.bytes;push('x column qdata',qc.q,scratch,'E4M3','[256, 256]','scratch','Recreated for W13 WGRAD');push('x column scales',qc.s,scratch+qc.q,'E8M0','[256, 8] logical transpose scales','scratch','Recreated with the column quantizer');add('h1',r*2*c.F*2,'BF16','[256, 512]','scratch','Replayed W13');scratch-=hf.bytes;push('h2 column qdata',hf.q,scratch,'E4M3','[256, 256]','scratch','Recreated for W2 WGRAD');push('h2 column scales',hf.s,scratch+hf.q,'E8M0','[256, 8] logical transpose scales','scratch','Recreated with SwiGLU quantization');}
    }
    return {c,r,slot,slotBytes,start,end,items,slotUsed:cursor-start,scratchUsed:end-scratch,base:0x10000000,mode,precision};
  }
  const schedule=['0F0','0F1','2F0','2F1','0F2','2B0','0F3','2B1','2F2','0B0','2F3','0F4','2B2','0B1','2F4','0F5','2B3','0B2','2F5','0F6','2B4','0B3','2F6','0F7','2B5','0B4','2F7','2B6','0B5','2B7','0B6','0B7'];
  function lifetimes(mode) {
    const map=new Map();schedule.forEach((a,p)=>{const stage=Number(a[0]),kind=a[1],mb=Number(a.slice(2)),key=mode==='stage'?`${stage},${mb}`:`${mb}`;if(kind==='F'){if(!map.has(key))map.set(key,{key,start:p,end:null,mb,stage});}else map.get(key).end=p+1;});
    const list=[...map.values()].sort((a,b)=>a.start-b.start);
    const counts=schedule.map((_,p)=>list.filter(iv=>iv.start<=p&&iv.end>p+1).length);
    const peak=Math.max(0,...counts),peakPositions=counts.flatMap((n,p)=>n===peak?[p]:[]);
    return {list,counts,peak,peakPositions};
  }
  function intervals(mode) {
    const list=lifetimes(mode).list.map(iv=>({...iv})),ends=[];
    for(const iv of list){let slot=ends.findIndex(e=>e<=iv.start);if(slot<0)slot=ends.length;ends[slot]=iv.end;iv.slot=slot;}
    return {list,slots:ends.length,depth:mode==='stage'?2:4,layerBytes:16*2**20};
  }
  function fp8Value(code) {const sign=code&128?-1:1,u=code&127,e=u>>3,m=u&7;if(e===15&&m===7)return NaN;return sign*(e===0?m*2**-9:(1+m/8)*2**(e-7));}
  function fp8Byte(value) {const sign=value<0||Object.is(value,-0)?128:0,v=Math.abs(value);let best=0,error=Infinity;for(let b=0;b<=126;b++){const e=Math.abs(fp8Value(b)-v);if(e<error||(e===error&&b%2===0)){best=b;error=e;}}return best|sign;}
  function quantBlock(outlier=false) {const seed=[-13.25,-7.5,-3.5625,-1.3125,-.234375,0,.005859375,.0009765625,2.75,3.625,6.75,9.5,11.25,13.5,15.75,17.25];const x=[...seed,...seed.map(v=>v*.75)];if(outlier)x[31]=512;const amax=Math.max(...x.map(Math.abs)),exponent=Math.ceil(Math.log2(amax/448)),scale=2**exponent,scaleByte=exponent+127;const bytes=x.map(v=>fp8Byte(v/scale)),q=bytes.map(fp8Value),decoded=q.map(v=>v*scale);return {x,amax,scale,scaleByte,bytes,q,decoded};}
  return {align,sum,packing,row,col,capacity,savedParts,scratchParts,memoryBounds,slotFromFactor,factorForSlot,memory,tokenData,routingRows,layout,schedule,lifetimes,intervals,fp8Value,fp8Byte,quantBlock};
})();
