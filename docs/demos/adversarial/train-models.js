// Train the six MNIST networks used by these demos, with the same advcore.js the pages run.
// Usage: put the four MNIST .gz files in this folder, then: node train-models.js 30
// It writes models.bin (int8 weights, one scale per output unit) and models.json; the pages
// load them split into models/<id>.bin with models/index.json.
require('./advcore.js');
const fs=require('fs'),zlib=require('zlib');
const TR=zlib.gunzipSync(fs.readFileSync('train-images-idx3-ubyte.gz')).subarray(16),TL=zlib.gunzipSync(fs.readFileSync('train-labels-idx1-ubyte.gz')).subarray(8);
const TE=zlib.gunzipSync(fs.readFileSync('t10k-images-idx3-ubyte.gz')).subarray(16),EL=zlib.gunzipSync(fs.readFileSync('t10k-labels-idx1-ubyte.gz')).subarray(8);
const X=[],Y=[];for(let i=0;i<12000;i++){X.push(Float32Array.from(TR.subarray(i*784,(i+1)*784),v=>v/255));Y.push(TL[i]);}
for(let i=0;i<2000;i++){X.push(Float32Array.from(TE.subarray(i*784,(i+1)*784),v=>v/255));Y.push(EL[i]);}
const ALL=[...Array(12000).keys()],P1=ALL.slice(0,6000),P2=ALL.slice(6000),TEST=[...Array(2000).keys()].map(i=>12000+i);
const EP=+process.argv[2]||30;
const MODELS=[
 {id:'FC100',name:'FC100-100-10',spec:{sizes:[784,100,100,10],act:'sigmoid',lambda:[1e-5,1e-5,1e-6],seed:2},data:'all'},
 {id:'FC50',name:'FC50-50-10',spec:{sizes:[784,50,50,10],act:'sigmoid',lambda:[1e-5,1e-5,1e-6],seed:3},data:'all'},
 {id:'FC10',name:'FC10 (softmax)',spec:{sizes:[784,10],act:'sigmoid',lambda:[1e-4],seed:1},data:'all'},
 {id:'P1a',name:'FC100-100-10, half 1',spec:{sizes:[784,100,100,10],act:'sigmoid',lambda:[1e-5,1e-5,1e-6],seed:5},data:'P1'},
 {id:'P1b',name:'FC100-100-10, half 1, new seed',spec:{sizes:[784,100,100,10],act:'sigmoid',lambda:[1e-5,1e-5,1e-6],seed:6},data:'P1'},
 {id:'P2',name:'FC100-100-10, half 2',spec:{sizes:[784,100,100,10],act:'sigmoid',lambda:[1e-5,1e-5,1e-6],seed:7},data:'P2'},
];
const out={models:[],note:'int8 weights, one float scale per output unit; biases float32'};const bufs=[];let off=0;
for(const m of MODELS){const idx=m.data==='all'?ALL:m.data==='P1'?P1:P2;const net=ADV.makeMLP(m.spec),r=ADV.rng(m.spec.seed+100);const t0=Date.now();
 for(let ep=0;ep<EP;ep++){const o=idx.slice();for(let i=o.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[o[i],o[j]]=[o[j],o[i]];}
  const lr=0.1*(ep<EP*0.7?1:0.3);for(let b=0;b<o.length;b+=32)ADV.trainBatch(net,X,Y,o.slice(b,b+32),lr,0.9);}
 const accF=ADV.accuracy(net,X,Y,TEST);
 // quantize
 const layers=[];
 net.layers.forEach(ly=>{const q=new Int8Array(ly.W.length),sc=new Float32Array(ly.nout);
  for(let j=0;j<ly.nout;j++){let mx=1e-12;for(let i=0;i<ly.nin;i++)mx=Math.max(mx,Math.abs(ly.W[j*ly.nin+i]));sc[j]=mx/127;for(let i=0;i<ly.nin;i++){const v=Math.round(ly.W[j*ly.nin+i]/sc[j]);q[j*ly.nin+i]=v;ly.W[j*ly.nin+i]=v*sc[j];}}
  layers.push({nin:ly.nin,nout:ly.nout,w:off,scale:off+q.length,b:off+q.length+4*ly.nout});
  bufs.push(Buffer.from(q.buffer),Buffer.from(sc.buffer),Buffer.from(ly.b.buffer));off+=q.length+8*ly.nout;
  const pad=(4-off%4)%4;if(pad){bufs.push(Buffer.alloc(pad));off+=pad;}});
 const accQ=ADV.accuracy(net,X,Y,TEST),accTr=ADV.accuracy(net,X,Y,idx.slice(0,3000));
 out.models.push({id:m.id,name:m.name,act:m.spec.act,lambda:m.spec.lambda,data:m.data,sizes:m.spec.sizes,layers,testAcc:+accQ.toFixed(4),trainAcc:+accTr.toFixed(4)});
 console.log(m.id,'test',accF.toFixed(4),'quantized',accQ.toFixed(4),'train',accTr.toFixed(4),'time',(Date.now()-t0)/1000);}
fs.writeFileSync('models.bin',Buffer.concat(bufs));fs.writeFileSync('models.json',JSON.stringify(out));
console.log('bytes',off);
