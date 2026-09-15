// Headless invariant tests for Transformer Forge.
// Loads the <script id="engine"> block in a vm sandbox (no DOM) and checks
// the algorithm engine the way the skill mandates.
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if(!m){ console.error('engine script not found'); process.exit(1); }
const ctx={console,Math,Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx; vm.createContext(ctx); vm.runInContext(m[1],ctx,{filename:'engine.js'});
const TRNS=ctx.TRNS;

let pass=0,fail=0,fails=[];
function ok(name,cond,extra){ if(cond){pass++;} else {fail++; fails.push(name+(extra?(' | '+extra):''));} }

// ---------- 1. softmax sums to 1 ----------
(function(){
  let allOne=true, rng=TRNS.mulberry32(1);
  for(let k=0;k<200;k++){
    let v=[]; for(let i=0;i<10;i++) v.push(rng()*6-3);
    let s=TRNS.softmax(v); let sum=0; for(const x of s) sum+=x;
    if(Math.abs(sum-1)>1e-12) allOne=false;
    if(!s.every(Number.isFinite)) allOne=false;
  }
  ok('softmax sums to 1 & finite', allOne);
})();

// ---------- 2. sinusoidal PE exact values + periodicity ----------
(function(){
  const d=8, pe=TRNS.sinusoidalPE(12,d);
  const s1=Math.sin(1), c1=Math.cos(1), c01=Math.cos(1/Math.pow(10000,2*1/8));
  ok('PE[0][0]=sin(0)=0', Math.abs(pe[0][0]-0)<1e-12, pe[0][0]);
  ok('PE[0][1]=cos(0)=1', Math.abs(pe[0][1]-1)<1e-12, pe[0][1]);
  ok('PE[1][0]=sin(1)', Math.abs(pe[1][0]-s1)<1e-12, pe[1][0]);
  ok('PE[1][1]=cos(1/denom)', Math.abs(pe[1][1]-c01)<1e-12, pe[1][1]);
  // PE is continuous & bounded in [-1,1]
  let bounded=true; for(const row of pe) for(const x of row) if(x<-1.0001||x>1.0001) bounded=false;
  ok('PE bounded in [-1,1]', bounded);
})();

// ---------- 3. attention: row-sum = 1 & strict causal mask (j>i = 0) ----------
(function(){
  const model=TRNS.buildModel({L:3,C:5,dModel:16,nHeads:2,layers:1,dFF:32,seed:11});
  const ex=TRNS.makeData(TRNS.mulberry32(5),1,3,5)[0];
  const c=TRNS.forward(model, ex.inputs);
  const A=c.bc[c.bc.length-1].aC.A[0]; // head 0
  let rowOk=true, causalOk=true, T=A.length;
  for(let i=0;i<T;i++){
    let s=0; for(let j=0;j<T;j++){ s+=A[i][j]; if(j>i && A[i][j]!==0) causalOk=false; }
    if(Math.abs(s-1)>1e-9) rowOk=false;
  }
  ok('attention row-sum = 1', rowOk);
  ok('attention strictly causal (A[i][j]=0 for j>i)', causalOk);
})();

// ---------- 4. forward on a single token does not NaN ----------
(function(){
  const model=TRNS.buildModel({L:3,C:5,dModel:16,nHeads:2,layers:1,dFF:32,seed:9});
  const c=TRNS.forward(model,[2]);
  let finite=true; for(const row of c.logits) for(const v of row) if(!isFinite(v)) finite=false;
  ok('single-token forward finite (T=1, no NaN)', finite);
})();

// ---------- 5. gradient check: analytic vs central finite difference ----------
(function(){
  const model=TRNS.buildModel({L:2,C:3,dModel:8,nHeads:2,layers:1,dFF:16,seed:7});
  const ex=TRNS.makeData(TRNS.mulberry32(3),1,2,3)[0];
  const r0=TRNS.lossAndGrads(model,[ex]);
  function loss(){ return TRNS.lossAndGrads(model,[ex]).loss; }
  let sp=[];
  (function rec(node,path){
    if(Array.isArray(node)){ if(node.length&&typeof node[0]==='number'){ for(let i=0;i<node.length;i++) sp.push({arr:node,idx:i,path:path}); } else { for(let i=0;i<node.length;i++) rec(node[i],path+'.'+i); } }
    else if(node&&typeof node==='object'){ for(const k in node) rec(node[k],path+'.'+k); }
  })(model.p,'');
  const eps=1e-5; let maxRel=0;
  for(const s of sp){
    let g=r0.grads.p; for(const p of s.path.split('.').filter(Boolean)) g=g[p];
    const a=s.arr[s.idx];
    s.arr[s.idx]=a+eps; const lp=loss(); s.arr[s.idx]=a-eps; const lm=loss(); s.arr[s.idx]=a;
    const num=(lp-lm)/(2*eps);
    const rel=Math.abs(g[s.idx]-num)/(Math.max(Math.abs(g[s.idx]),Math.abs(num),1e-6));
    if(rel>maxRel) maxRel=rel;
  }
  ok('gradient check maxRel < 1e-4', maxRel<1e-4, 'maxRel='+maxRel.toExponential(2)+' scalars='+sp.length);
})();

// ---------- 6. training determinism (same seed => bit-identical history) ----------
(function(){
  const cfg={L:2,C:4,dModel:12,nHeads:2,layers:1,dFF:24,iters:30,batch:16,lr:0.003,seed:42};
  const a=TRNS.train(cfg), b=TRNS.train(cfg);
  ok('training deterministic (bit-identical hist)', JSON.stringify(a.hist)===JSON.stringify(b.hist));
})();

// ---------- 7. training converges: loss drops + echo learned ----------
(function(){
  const cfg={L:3,C:5,dModel:16,nHeads:2,layers:1,dFF:32,iters:150,batch:16,lr:0.003,seed:42};
  const res=TRNS.train(cfg);
  const first=res.hist[0], last=res.hist[res.hist.length-1];
  ok('loss decreases', first>last*1.15, first.toFixed(2)+'->'+last.toFixed(2));
  ok('echo accuracy > 2x random (random=0.20)', res.evalAcc>0.40, (res.evalAcc*100).toFixed(0)+'%');
  ok('at least some sequences fully echoed', res.fullMatch>=1, res.fullMatch+'/'+res.evalN);
})();

// ---------- 8. generate reproduces the input content after DELIM ----------
(function(){
  const cfg={L:3,C:5,dModel:16,nHeads:2,layers:1,dFF:32,iters:150,batch:16,lr:0.003,seed:42};
  const res=TRNS.train(cfg);
  let okCount=0; const rng=TRNS.mulberry32(777);
  for(let i=0;i<10;i++){
    const content=[]; for(let j=0;j<3;j++) content.push((rng()*5)|0);
    const prefix=content.slice(); prefix.push(5); // DELIM=C
    const gen=TRNS.generate(res.model,prefix,3,rng);
    let all=true; for(let t=0;t<3;t++) if(gen[t]!==content[t]) all=false;
    if(all) okCount++;
  }
  ok('generate echoes content (>=8/10 exact)', okCount>=8, okCount+'/10');
})();

const report=`PASS ${pass} / ${pass+fail}\n`+(fail?('FAIL '+fails.join(' ; ')):'ALL GREEN')+'\n';
fs.writeFileSync(path.join(__dirname,'_smoke.log'), report);
console.log(report);
process.exit(fail?1:0);
