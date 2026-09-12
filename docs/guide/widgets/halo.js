var W=[0.906,0.423],G=0.32;
var B=[[0.4,-0.1],[0.9,0.1],[0.5,0.45],[0.75,-0.5],[0.25,0.8],[1.0,0.3],[0.8,-0.75],[0.45,0.15]];
var P=[];for(var q=0;q<B.length;q++){P.push([B[q][0],B[q][1],1]);P.push([-B[q][0],-B[q][1],-1]);}
var w,k,idx,hit,hs,phase,msg;
function X(a){return (340+115*a).toFixed(1);}
function Y(b){return (200-115*b).toFixed(1);}
function seg(p,r,st){return '<line x1="'+X(p[0])+'" y1="'+Y(p[1])+'" x2="'+X(r[0])+'" y2="'+Y(r[1])+'" '+st+'/>';}
function bad(p){return p[2]*(w[0]*p[0]+w[1]*p[1])<=0;}
function render(){
 var u=[-W[1],W[0]],t=1.14,s='',n=Math.hypot(w[0],w[1]);
 if(n>0){var d=[w[0]/n,w[1]/n],v=[-d[1],d[0]];
  s+='<polygon clip-path="url(#pc)" points="'+X(2*v[0])+','+Y(2*v[1])+' '+X(-2*v[0])+','+Y(-2*v[1])+' '+X(-2*v[0]+2.8*d[0])+','+Y(-2*v[1]+2.8*d[1])+' '+X(2*v[0]+2.8*d[0])+','+Y(2*v[1]+2.8*d[1])+'" fill="#378ADD" opacity="0.10"/>';}
 var c1=[G*W[0]+t*u[0],G*W[1]+t*u[1]],c2=[G*W[0]-t*u[0],G*W[1]-t*u[1]],c3=[-G*W[0]-t*u[0],-G*W[1]-t*u[1]],c4=[-G*W[0]+t*u[0],-G*W[1]+t*u[1]];
 s+='<polygon points="'+X(c1[0])+','+Y(c1[1])+' '+X(c2[0])+','+Y(c2[1])+' '+X(c3[0])+','+Y(c3[1])+' '+X(c4[0])+','+Y(c4[1])+'" fill="#1D9E75" opacity="0.12"/>';
 s+=seg([t*u[0],t*u[1]],[-t*u[0],-t*u[1]],'stroke="#0F6E56" stroke-width="1" stroke-dasharray="5 4"');
 s+=seg([0,0],[0.6*W[0],0.6*W[1]],'stroke="#0F6E56" stroke-width="2" marker-end="url(#arrow)"');
 s+='<text class="ts" x="'+X(0.66*W[0])+'" y="'+Y(0.66*W[1])+'" fill="#0F6E56">w* (true north)</text>';
 s+='<text class="ts" x="'+X(c1[0])+'" y="'+(Y(c1[1])-8)+'" text-anchor="middle" fill="#0F6E56">margin corridor &gamma;</text>';
 var nb=0;
 for(var i=0;i<P.length;i++){var p=P[i];
  if(bad(p)){nb++;s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="9.5" fill="none" stroke="#E24B4A" stroke-width="1" opacity="0.55"/>';}
  s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="5.5" fill="'+(p[2]>0?'#378ADD':'var(--surface-2)')+'" stroke="'+(p[2]>0?'#185FA5':'#5F5E5A')+'" stroke-width="1"/>';}
 if(n>0){var d2=[w[0]/n,w[1]/n],v2=[-d2[1],d2[0]];
  s+=seg([1.14*v2[0],1.14*v2[1]],[-1.14*v2[0],-1.14*v2[1]],'stroke="#BA7517" stroke-width="2"');
  s+=seg([0,0],[0.5*d2[0],0.5*d2[1]],'stroke="#BA7517" stroke-width="2" marker-end="url(#arrow)"');
  s+='<text class="ts" x="'+X(0.58*d2[0])+'" y="'+Y(0.58*d2[1])+'" fill="#854F0B">w</text>';
  s+='<text class="ts" x="'+X(1.75*d2[0])+'" y="'+Y(1.75*d2[1])+'" text-anchor="middle" fill="#185FA5">w says +</text>';}
 if(hit!=null){var h=P[hit];
  s+='<circle cx="'+X(h[0])+'" cy="'+Y(h[1])+'" r="13" fill="none" stroke="'+(hs==='bad'?'#E24B4A':'#1D9E75')+'" stroke-width="2.5"/>';}
 document.getElementById('stage').innerHTML=s;
 document.getElementById('readout').innerHTML=msg+'<br>updates applied <b>k = '+k+'</b> &nbsp;&middot;&nbsp; currently misclassified <b>'+nb+' of '+P.length+'</b>';
}
function act(){var btn=document.getElementById('go');
 if(phase==='find'){
  for(var j=0;j<P.length;j++){var i=(idx+j)%P.length,p=P[i];
   if(bad(p)){hit=i;hs='bad';phase='apply';btn.textContent='Apply update';
    msg='This one is labeled '+(p[2]>0?'+ but lies outside the shaded region':'&minus; but lies inside the shaded region')+'.';
    render();return;}}
  hit=null;msg='No mistakes left: every point is on its own side, and w has stopped changing.';render();return;}
 var g=P[hit];w=[w[0]+g[2]*g[0],w[1]+g[2]*g[1]];k++;idx=hit+1;hs='fixed';phase='find';
 btn.textContent='Find a mistake';
 msg='Boundary swung past it. Check what that did to the others.';
 render();}
function reset(){w=[-1.4,2.0];k=0;idx=0;hit=null;hs=null;phase='find';
 msg='w starts pointed the wrong way. Every haloed point is currently misclassified.';
 document.getElementById('go').textContent='Find a mistake';render();}
reset();
