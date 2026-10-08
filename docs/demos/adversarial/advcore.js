// Shared core for the adversarial-examples demos: a small MLP on MNIST, trained
// with minibatch SGD, plus the gradient of the loss with respect to the INPUT,
// the paper's penalty-method search for a minimal adversarial perturbation, and
// operator norms for the spectral bound.  Runs in the browser and in node.
(function(root){
'use strict';
function rng(seed){var a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function randn(r){var u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
// spec: {sizes:[784,100,100,10], act:'sigmoid'|'relu', lambda:[...per layer], seed}
function makeMLP(spec){
  var r=rng(spec.seed||1),L=[];
  for(var l=0;l+1<spec.sizes.length;l++){var nin=spec.sizes[l],nout=spec.sizes[l+1],W=new Float32Array(nin*nout),b=new Float32Array(nout);
    var sd=spec.act==='relu'?Math.sqrt(2/nin):Math.sqrt(1/nin);
    for(var i=0;i<W.length;i++)W[i]=randn(r)*sd;
    L.push({nin:nin,nout:nout,W:W,b:b,vW:new Float32Array(W.length),vb:new Float32Array(nout)});}
  return {spec:spec,layers:L,act:spec.act||'sigmoid',lambda:spec.lambda||L.map(function(){return 0;})};}
function actf(a,z){return a==='relu'?(z>0?z:0):1/(1+Math.exp(-z));}
function actd(a,y){return a==='relu'?(y>0?1:0):y*(1-y);}
// forward: returns list of activations [x, h1, ..., probs]
function forward(net,x){
  var acts=[x],h=x,L=net.layers;
  for(var l=0;l<L.length;l++){var ly=L[l],o=new Float32Array(ly.nout),last=l===L.length-1;
    for(var j=0;j<ly.nout;j++){var s=ly.b[j],row=j*ly.nin;for(var i=0;i<ly.nin;i++)s+=ly.W[row+i]*h[i];o[j]=s;}
    if(last){var m=-1e9;for(j=0;j<o.length;j++)if(o[j]>m)m=o[j];var z=0;for(j=0;j<o.length;j++){o[j]=Math.exp(o[j]-m);z+=o[j];}for(j=0;j<o.length;j++)o[j]/=z;}
    else for(j=0;j<o.length;j++)o[j]=actf(net.act,o[j]);
    acts.push(o);h=o;}
  return acts;}
function predict(net,x){var p=forward(net,x);p=p[p.length-1];var b=0;for(var j=1;j<p.length;j++)if(p[j]>p[b])b=j;return b;}
// backward from d(loss)/d(logits) = p - onehot(y).  If grads is given, accumulates
// weight gradients into it.  Always returns d(loss)/d(input): the same chain rule,
// carried one layer further, to the pixels.
function backward(net,acts,y,grads){
  var L=net.layers,p=acts[acts.length-1],d=new Float32Array(p.length);
  for(var j=0;j<p.length;j++)d[j]=p[j]-(j===y?1:0);
  for(var l=L.length-1;l>=0;l--){var ly=L[l],h=acts[l],dh=new Float32Array(ly.nin);
    if(grads){var gW=grads[l].W,gb=grads[l].b;for(j=0;j<ly.nout;j++){var dj=d[j];if(dj===0)continue;gb[j]+=dj;var row=j*ly.nin;for(var i=0;i<ly.nin;i++)gW[row+i]+=dj*h[i];}}
    for(j=0;j<ly.nout;j++){dj=d[j];if(dj===0)continue;row=j*ly.nin;for(i=0;i<ly.nin;i++)dh[i]+=dj*ly.W[row+i];}
    if(l>0)for(i=0;i<ly.nin;i++)dh[i]*=actd(net.act,h[i]);
    d=dh;}
  return d;}
function zeroGrads(net){return net.layers.map(function(ly){return {W:new Float32Array(ly.W.length),b:new Float32Array(ly.nout)};});}
// one minibatch of SGD with momentum and weight decay
function trainBatch(net,X,Y,idx,lr,mom){
  var g=zeroGrads(net),loss=0;
  for(var k=0;k<idx.length;k++){var a=forward(net,X[idx[k]]),p=a[a.length-1];loss-=Math.log(Math.max(p[Y[idx[k]]],1e-12));backward(net,a,Y[idx[k]],g);}
  var n=idx.length;
  net.layers.forEach(function(ly,l){var lam=net.lambda[l]||0;
    for(var i=0;i<ly.W.length;i++){var gi=g[l].W[i]/n+lam*ly.W[i];ly.vW[i]=mom*ly.vW[i]-lr*gi;ly.W[i]+=ly.vW[i];}
    for(i=0;i<ly.nout;i++){gi=g[l].b[i]/n;ly.vb[i]=mom*ly.vb[i]-lr*gi;ly.b[i]+=ly.vb[i];}});
  return loss/n;}
function accuracy(net,X,Y,idx){var ok=0;for(var k=0;k<idx.length;k++)if(predict(net,X[idx[k]])===Y[idx[k]])ok++;return ok/idx.length;}
// gradient of the loss for label y with respect to the input image
function inputGrad(net,x,y){var a=forward(net,x);return {g:backward(net,a,y,null),p:a[a.length-1]};}
// The paper's search: minimize c|r| + loss(x+r, target) subject to x+r in [0,1]^n,
// for the smallest c that still reaches the target.  Here |r| is squared L2 and the
// inner problem is solved by projected gradient descent instead of L-BFGS.
function attackFixedC(net,x,target,c,steps,lr){
  var n=x.length,r=new Float32Array(n),xr=new Float32Array(n);
  for(var s=0;s<steps;s++){
    for(var i=0;i<n;i++)xr[i]=x[i]+r[i];
    var gr=inputGrad(net,xr,target).g;
    for(i=0;i<n;i++){var v=r[i]-lr*(gr[i]+2*c*r[i]);var xv=x[i]+v;if(xv<0)v=-x[i];else if(xv>1)v=1-x[i];r[i]=v;}}
  for(i=0;i<n;i++)xr[i]=x[i]+r[i];
  return {r:r,xr:xr,ok:predict(net,xr)===target};}
function attack(net,x,target,opts){
  opts=opts||{};var steps=opts.steps||60,lr=opts.lr||0.5,best=null,c=opts.c0||1;
  // line search on c: shrink c until the attack succeeds, then refine by bisection
  var lo=0,hi=null;
  for(var it=0;it<(opts.iters||10);it++){var res=attackFixedC(net,x,target,c,steps,lr);
    if(res.ok){best=res;best.c=c;lo=c;c=hi===null?c*2:(c+hi)/2;}else{hi=c;c=(lo+c)/2;if(c<1e-6)break;}}
  if(!best){best=attackFixedC(net,x,target,0,steps*2,lr);best.c=0;}
  best.dist=distortion(best.r);return best;}
function distortion(r){var s=0;for(var i=0;i<r.length;i++)s+=r[i]*r[i];return Math.sqrt(s/r.length);}
// largest singular value of a layer's weight matrix, by power iteration
function opNorm(ly,iters){var r=rng(3),v=new Float32Array(ly.nin);for(var i=0;i<ly.nin;i++)v[i]=r()-0.5;var s=0;
  for(var it=0;it<(iters||40);it++){var u=new Float32Array(ly.nout);for(var j=0;j<ly.nout;j++){var a=0,row=j*ly.nin;for(i=0;i<ly.nin;i++)a+=ly.W[row+i]*v[i];u[j]=a;}
    var w=new Float32Array(ly.nin);for(j=0;j<ly.nout;j++){row=j*ly.nin;for(i=0;i<ly.nin;i++)w[i]+=ly.W[row+i]*u[j];}
    var nr=0;for(i=0;i<ly.nin;i++)nr+=w[i]*w[i];nr=Math.sqrt(nr);for(i=0;i<ly.nin;i++)v[i]=w[i]/nr;s=Math.sqrt(nr);}
  return s;}
root.ADV={rng:rng,randn:randn,makeMLP:makeMLP,forward:forward,predict:predict,backward:backward,trainBatch:trainBatch,accuracy:accuracy,
  inputGrad:inputGrad,attack:attack,attackFixedC:attackFixedC,distortion:distortion,opNorm:opNorm};
})(typeof window!=='undefined'?window:global);
// ---- browser loaders: 1,000 held-out MNIST test digits, and the trained networks, one file each
(function(root){
if(typeof window==='undefined')return;
root.ADV.loadMNIST=function(base){base=base||'';
  return Promise.all([new Promise(function(res,rej){var im=new Image();im.onload=function(){res(im);};im.onerror=rej;im.src=base+'mnist-test1k.png';}),
    fetch(base+'mnist-test1k-labels.bin').then(function(r){return r.arrayBuffer();})]).then(function(a){
    var im=a[0],cv=document.createElement('canvas');cv.width=im.width;cv.height=im.height;var cx=cv.getContext('2d');cx.drawImage(im,0,0);
    var px=cx.getImageData(0,0,im.width,im.height).data,cols=im.width/28,Y=new Uint8Array(a[1]),X=[];
    for(var n=0;n<Y.length;n++){var r0=Math.floor(n/cols)*28,c0=(n%cols)*28,x=new Float32Array(784);
      for(var i=0;i<28;i++)for(var j=0;j<28;j++)x[i*28+j]=px[((r0+i)*im.width+c0+j)*4]/255;X.push(x);}
    return {X:X,Y:Array.from(Y),train:[],test:Array.from({length:Y.length},function(_,i){return i;})};});};
var INDEX=null,CACHE={};
root.ADV.modelIndex=function(base){base=base||'';if(!INDEX)INDEX=fetch(base+'models/index.json').then(function(r){return r.json();}).then(function(j){return j.models;});return INDEX;};
root.ADV.loadModel=function(base,id){base=base||'';if(CACHE[id])return CACHE[id];
  CACHE[id]=root.ADV.modelIndex(base).then(function(list){var m=list.filter(function(q){return q.id===id;})[0];
    return fetch(base+m.file).then(function(r){return r.arrayBuffer();}).then(function(buf){
      var net={spec:{sizes:m.sizes},act:m.act,lambda:m.lambda,meta:m,layers:[]};
      m.layers.forEach(function(L){var q=new Int8Array(buf,L.w,L.nin*L.nout),sc=new Float32Array(buf,L.scale,L.nout),b=new Float32Array(buf,L.b,L.nout).slice(),W=new Float32Array(L.nin*L.nout);
        for(var j=0;j<L.nout;j++)for(var i=0;i<L.nin;i++)W[j*L.nin+i]=q[j*L.nin+i]*sc[j];
        net.layers.push({nin:L.nin,nout:L.nout,W:W,b:b});});
      return net;});});
  return CACHE[id];};
root.ADV.loadModels=function(base,ids){return Promise.all(ids.map(function(id){return root.ADV.loadModel(base,id);})).then(function(nets){var o={};ids.forEach(function(id,k){o[id]=nets[k];});return o;});};
})(typeof window!=="undefined"?window:global);
