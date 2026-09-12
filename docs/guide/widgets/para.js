var W=[0.906,0.423],G=0.32,S=115;
var B=[[0.4,-0.1],[0.9,0.1],[0.5,0.45],[0.75,-0.5],[0.25,0.8],[1.0,0.3],[0.8,-0.75],[0.45,0.15]];
var P=[];for(var q=0;q<B.length;q++){P.push([B[q][0],B[q][1],1]);P.push([-B[q][0],-B[q][1],-1]);}
var w,k,idx,hit,hs,phase,msg,wOld,cor;
function X(a){return (340+S*a).toFixed(1);}
function Y(b){return (200-S*b).toFixed(1);}
function seg(p,r,st){return '<line x1="'+X(p[0])+'" y1="'+Y(p[1])+'" x2="'+X(r[0])+'" y2="'+Y(r[1])+'" '+st+'/>';}
function bad(p){return p[2]*(w[0]*p[0]+w[1]*p[1])<=0;}
function render(){
 var u=[-W[1],W[0]],t=1.14,s='',n=Math.hypot(w[0],w[1]);
 if(n>0){var d=[w[0]/n,w[1]/n],v=[-d[1],d[0]];
  s+='<polygon clip-path="url(#pc)" points="'+X(2*v[0])+','+Y(2*v[1])+' '+X(-2*v[0])+','+Y(-2*v[1])+' '+X(-2*v[0]+2.8*d[0])+','+Y(-2*v[1]+2.8*d[1])+' '+X(2*v[0]+2.8*d[0])+','+Y(2*v[1]+2.8*d[1])+'" fill="#378ADD" opacity="0.09"/>';}
 var c1=[G*W[0]+t*u[0],G*W[1]+t*u[1]],c2=[G*W[0]-t*u[0],G*W[1]-t*u[1]],c3=[-G*W[0]-t*u[0],-G*W[1]-t*u[1]],c4=[-G*W[0]+t*u[0],-G*W[1]+t*u[1]];
 s+='<polygon points="'+X(c1[0])+','+Y(c1[1])+' '+X(c2[0])+','+Y(c2[1])+' '+X(c3[0])+','+Y(c3[1])+' '+X(c4[0])+','+Y(c4[1])+'" fill="#1D9E75" opacity="0.12"/>';
 s+=seg([t*u[0],t*u[1]],[-t*u[0],-t*u[1]],'stroke="#0F6E56" stroke-width="1" stroke-dasharray="5 4"');
 s+=seg([0,0],[0.55*W[0],0.55*W[1]],'stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"');
 s+='<text class="ts" x="'+X(0.62*W[0])+'" y="'+Y(0.62*W[1])+'" fill="#0F6E56">w*</text>';
 var nb=0;
 for(var i=0;i<P.length;i++){var p=P[i];
  if(bad(p)){nb++;s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="9.5" fill="none" stroke="#E24B4A" stroke-width="1" opacity="0.5"/>';}
  s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="5" fill="'+(p[2]>0?'#378ADD':'var(--surface-2)')+'" stroke="'+(p[2]>0?'#185FA5':'#5F5E5A')+'" stroke-width="1"/>';}
 if(hs==='bad'&&hit!=null){var h=P[hit],c=[h[2]*h[0],h[2]*h[1]];
  s+='<circle cx="'+X(h[0])+'" cy="'+Y(h[1])+'" r="12" fill="none" stroke="#E24B4A" stroke-width="2.5"/>';
  if(h[2]<0)s+=seg([h[0],h[1]],c,'stroke="#D85A30" stroke-width="0.5" stroke-dasharray="3 3"');
  s+=seg([0,0],c,'stroke="#D85A30" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arrow)"');
  s+='<text class="ts" x="'+X(c[0]*0.55+0.12)+'" y="'+Y(c[1]*0.55-0.07)+'" fill="#993C1D">y&#183;x</text>';}
 if(hs==='fixed'&&hit!=null){
  s+=seg([0,0],cor,'stroke="#D85A30" stroke-width="2" marker-end="url(#arrow)"');
  s+=seg([0,0],wOld,'stroke="#BA7517" stroke-width="1.5" opacity="0.5"');
  s+=seg(wOld,w,'stroke="#D85A30" stroke-width="0.75" stroke-dasharray="5 4" opacity="0.8"');
  s+=seg(cor,w,'stroke="#BA7517" stroke-width="0.75" stroke-dasharray="5 4" opacity="0.8"');
  s+='<text class="ts" x="'+X(cor[0]*0.5+0.1)+'" y="'+Y(cor[1]*0.5-0.06)+'" fill="#993C1D">y&#183;x</text>';
  s+='<text class="ts" x="'+X(wOld[0]*0.62-0.1)+'" y="'+Y(wOld[1]*0.62)+'" text-anchor="end" fill="#854F0B" opacity="0.7">w before</text>';}
 var nn=Math.hypot(w[0],w[1]);
 if(nn>0){var d2=[w[0]/nn,w[1]/nn],v2=[-d2[1],d2[0]];
  s+=seg([1.14*v2[0],1.14*v2[1]],[-1.14*v2[0],-1.14*v2[1]],'stroke="#BA7517" stroke-width="2"');
  s+=seg([0,0],w,'stroke="#BA7517" stroke-width="2.5" marker-end="url(#arrow)"');
  s+='<text class="ts" x="'+X(w[0]+0.07)+'" y="'+Y(w[1]+0.06)+'" fill="#854F0B">'+(hs==='fixed'?'w + y&#183;x':'w')+'</text>';}
 document.getElementById('stage').innerHTML=s;
 document.getElementById('readout').innerHTML=msg+'<br>updates applied <b>k = '+k+'</b> &nbsp;&middot;&nbsp; misclassified <b>'+nb+' of '+P.length+'</b>';
}
function act(){var btn=document.getElementById('go');
 if(phase==='find'){
  for(var j=0;j<P.length;j++){var i=(idx+j)%P.length,p=P[i];
   if(bad(p)){hit=i;hs='bad';phase='apply';btn.textContent='Apply update';
    msg='Labeled '+(p[2]>0?'+ but outside the shaded region. The dashed arrow is the point itself: that is what gets added to w.':'&minus; but inside the shaded region. The dashed arrow is the point flipped through the origin: that is what gets added to w.');
    render();return;}}
  hit=null;hs=null;msg='No mistakes left. w has stopped moving.';render();return;}
 var g=P[hit];wOld=[w[0],w[1]];cor=[g[2]*g[0],g[2]*g[1]];
 w=[w[0]+cor[0],w[1]+cor[1]];k++;idx=hit+1;hs='fixed';phase='find';
 btn.textContent='Find a mistake';
 msg='Parallelogram closed: w before, plus y&#183;x, gives the diagonal. The boundary is perpendicular to that diagonal.';
 render();}
function reset(){w=[-0.7,1.0];k=0;idx=0;hit=null;hs=null;phase='find';
 msg='w starts pointed the wrong way: 4 of 16 correct. Every haloed point is misclassified.';
 document.getElementById('go').textContent='Find a mistake';render();}
reset();
