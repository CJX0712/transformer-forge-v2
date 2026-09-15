// DOM-stub UI check: loads the <script id="ui"> block against a fake DOM and
// clicks every control once. Catches wiring bugs that _smoke.js cannot reach.
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const engineMatch=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const uiMatch=html.match(/<script id="ui">([\s\S]*?)<\/script>/);

let pass=0,fail=0,fails=[];
function ok(name,cond,extra){ if(cond){pass++;} else {fail++; fails.push(name+(extra?(' | '+extra):''));} }

// --- fake DOM ---
function makeCtx2d(){
  const calls={};
  const c={
    strokeStyle:'', fillStyle:'', font:'', lineWidth:1,
    clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fillRect(){}, fillText(){},
  };
  ['clearRect','beginPath','moveTo','lineTo','stroke','fillRect','fillText'].forEach(k=>{
    const real=c[k]; c[k]=function(){ calls[k]=(calls[k]||0)+1; real.apply(c,arguments); };
  });
  c._calls=calls;
  return c;
}
const elements={};
function makeEl(id){
  const listeners={};
  const el={
    id, value:'0', textContent:'', innerHTML:'', disabled:false, style:{}, _listeners:listeners, _ctx:null,
    addEventListener(ev,fn){ listeners[ev]=fn; },
    _fire(ev){ if(listeners[ev]) listeners[ev](); },
    getContext(){ if(!this._ctx) this._ctx=makeCtx2d(); return this._ctx; },
    appendChild(){},
  };
  return el;
}
['L','C','dModel','nHeads','layers','dFF','iters','lr','batch','seed','train','status',
 'stLoss','stAcc','stFull','stTime','lossCanvas','attnCanvas','genIn','genOut'].forEach(id=>{
  elements[id]=makeEl(id);
});
// default small config so training is fast under the vm sandbox
elements.L.value='2'; elements.C.value='4'; elements.dModel.value='12'; elements.nHeads.value='2';
elements.layers.value='1'; elements.dFF.value='24'; elements.iters.value='6';
elements.lr.value='0.005'; elements.batch.value='4'; elements.seed.value='42';

const document={ getElementById(id){ return elements[id]; } };
const ctx2={ console, Math, Object, Array, JSON, isFinite, Infinity,
  document, Date, setTimeout, requestAnimationFrame(){return 1;}, cancelAnimationFrame(){},
  globalThis:{} };
ctx2.globalThis=ctx2;
vm.createContext(ctx2);
vm.runInContext(engineMatch[1], ctx2, {filename:'engine.js'});
// wrap in a function so the UI's top-level `return` guard is legal under vm
vm.runInContext('(function(){'+uiMatch[1]+'})();', ctx2, {filename:'ui.js'});

// fire the train button
elements.train._fire('click');

// training runs inside setTimeout(20); wait, then assert
setTimeout(function(){
  try {
    ok('status updated after train', elements.status.textContent.indexOf('完成')>=0, elements.status.textContent);
    ok('loss stat populated', elements.stLoss.textContent!=='–' && elements.stLoss.textContent!=='', elements.stLoss.textContent);
    ok('acc stat populated', elements.stAcc.textContent!=='–' && elements.stAcc.textContent!=='', elements.stAcc.textContent);
    ok('fullMatch stat populated', elements.stFull.textContent!=='–', elements.stFull.textContent);
    ok('time stat populated', elements.stTime.textContent!=='–', elements.stTime.textContent);
    ok('gen input rendered', elements.genIn.innerHTML.length>0, elements.genIn.innerHTML);
    ok('gen output rendered', elements.genOut.innerHTML.length>0, elements.genOut.innerHTML);
    ok('loss canvas drawn (lineTo called)', (elements.lossCanvas._ctx._calls.lineTo||0)>0);
    ok('attn canvas drawn (fillRect called)', (elements.attnCanvas._ctx._calls.fillRect||0)>0);
  } catch(e){ ok('no exception during checks', false, e.message); }

  const report=`PASS ${pass} / ${pass+fail}\n`+(fail?('FAIL '+fails.join(' ; ')):'ALL GREEN')+'\n';
  fs.writeFileSync(path.join(__dirname,'_uicheck.log'), report);
  console.log(report);
  process.exit(fail?1:0);
}, 400);
