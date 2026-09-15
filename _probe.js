// ASCII dump of internal state to _probe.txt — the "read it, don't trust green" check.
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx={console,Math,Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx; vm.createContext(ctx); vm.runInContext(m[1],ctx,{filename:'engine.js'});
const TRNS=ctx.TRNS;

const out=[];
function log(s){ out.push(s); }

// train a small model
const res=TRNS.train({L:3,C:5,dModel:16,nHeads:2,layers:1,dFF:32,iters:150,batch:16,lr:0.003,seed:42});
log('=== Transformer Forge probe ===');
log('config: L=3 C=5 d=16 h=2 layers=1 dFF=32  iters=150  lr=0.003');
log('final loss = '+res.hist[res.hist.length-1].toFixed(3)+'   evalAcc = '+(res.evalAcc*100).toFixed(1)+'%   fullMatch = '+res.fullMatch+'/'+res.evalN);
log('loss curve (every 15): '+res.hist.filter((x,i)=>i%15===0).map(x=>x.toFixed(2)).join(' '));

// self-attention heatmap for one sample (head 0, last layer)
const samp=TRNS.makeData(TRNS.mulberry32(7),1,3,5)[0];
const c=TRNS.forward(res.model, samp.inputs);
const A=c.bc[c.bc.length-1].aC.A[0];
log('');
log('sample input tokens: '+samp.inputs.join(' ')+'   (8=DELIM)');
log('attention heatmap (head0, last layer) — rows=query, cols=key, . = ~0, # = ~1:');
let header='    '; for(let j=0;j<A.length;j++) header+=' '+String(j).padStart(2,' ');
log(header);
for(let i=0;i<A.length;i++){
  let row=String(i).padStart(2,' ')+' |';
  for(let j=0;j<A.length;j++){
    const v=A[i][j];
    const ch = v>0.66?'#': v>0.33?':': v>0.05?'.':' ';
    row+=' '+ch+' ';
  }
  log(row);
}
log('(causal: upper-right triangle should be blank)');

// echo examples
log('');
log('echo examples  content -> model output:');
for(let i=0;i<Math.min(6,res.evals.length);i++){
  const e=res.evals[i];
  log('  ['+e.content.join(',')+'] -> ['+e.gen.join(',')+']');
}
fs.writeFileSync(path.join(__dirname,'_probe.txt'), out.join('\n')+'\n');
console.log('probe written, '+out.length+' lines');
