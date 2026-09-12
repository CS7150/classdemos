var W=[0.906,0.423],G=0.323,R=1.14;
var P=[[0.9,0.1,1],[-0.6,-0.5,-1],[0.45,-0.2,1],[-0.8,0.4,-1],[0.35,0.75,1],[-0.45,0.2,-1],[0.8,-0.4,1],[-0.2,-0.9,-1],[0.6,0.5,1],[-1.0,-0.55,-1],[0.2,0.9,1],[-0.9,-0.1,-1],[1.0,0.55,1],[-0.35,-0.75,-1]];
var w,k,idx,hit,done;
function X(a){return (340+115*a).toFixed(1);}
function Y(b){return (200-115*b).toFixed(1);}
function seg(p,q,st){return '<line x1="'+X(p[0])+'" y1="'+Y(p[1])+'" x2="'+X(q[0])+'" y2="'+Y(q[1])+'" '+st+'/>';}
function render(){
 var u=[-W[1],W[0]],t=1.12,s='';
 var c1=[G*W[0]+t*u[0],G*W[1]+t*u[1]],c2=[G*W[0]-t*u[0],G*W[1]-t*u[1]];
 var c3=[-G*W[0]-t*u[0],-G*W[1]-t*u[1]],c4=[-G*W[0]+t*u[0],-G*W[1]+t*u[1]];
 s+='<polygon points="'+X(c1[0])+','+Y(c1[1])+' '+X(c2[0])+','+Y(c2[1])+' '+X(c3[0])+','+Y(c3[1])+' '+X(c4[0])+','+Y(c4[1])+'" fill="#1D9E75" opacity="0.10"/>';
 s+=seg([t*u[0],t*u[1]],[-t*u[0],-t*u[1]],'stroke="#0F6E56" stroke-width="1" stroke-dasharray="5 4"');
 s+=seg([0,0],[0.72*W[0],0.72*W[1]],'stroke="#0F6E56" stroke-width="2" marker-end="url(#arrow)"');
 s+='<text class="ts" x="'+X(0.78*W[0])+'" y="'+Y(0.78*W[1])+'" fill="#0F6E56">w* (true north)</text>';
 s+='<text class="ts" x="'+X(c1[0])+'" y="'+(Y(c1[1])-8)+'" text-anchor="middle" fill="#0F6E56">margin corridor &gamma;</text>';
 for(var i=0;i<P.length;i++){
  var p=P[i],f=p[2]>0?'#378ADD':'var(--surface-2)',st=p[2]>0?'#185FA5':'#5F5E5A';
  s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="5.5" fill="'+f+'" stroke="'+st+'" stroke-width="1"/>';
 }
 var n=Math.hypot(w[0],w[1]);
 if(n>0){
  var d=[w[0]/n,w[1]/n],v=[-d[1],d[0]];
  s+=seg([1.12*v[0],1.12*v[1]],[-1.12*v[0],-1.12*v[1]],'stroke="#BA7517" stroke-width="2"');
  s+=seg([0,0],[0.66*d[0],0.66*d[1]],'stroke="#BA7517" stroke-width="2" marker-end="url(#arrow)"');
  s+='<text class="ts" x="'+X(0.74*d[0])+'" y="'+Y(0.74*d[1])+'" fill="#854F0B">w</text>';
 }
 if(hit!=null){
  var h=P[hit];
  s+='<circle cx="'+X(h[0])+'" cy="'+Y(h[1])+'" r="11" fill="none" stroke="#E24B4A" stroke-width="1.5"/>';
  s+='<text class="ts" x="'+X(h[0])+'" y="'+(Y(h[1])+30)+'" text-anchor="middle" fill="#A32D2D">mistake</text>';
 }
 document.getElementById('stage').innerHTML=s;
 var sh=W[0]*w[0]+W[1]*w[1],ln=Math.hypot(w[0],w[1]);
 document.getElementById('readout').innerHTML=
  'mistakes so far <b>k = '+k+'</b><br>'+
  'shadow on w*: <b>'+sh.toFixed(2)+'</b> &nbsp;&ge;&nbsp; k&gamma; = '+(k*G).toFixed(2)+'<br>'+
  'length |w|: <b>'+ln.toFixed(2)+'</b> &nbsp;&le;&nbsp; &radic;k&middot;R = '+(Math.sqrt(k)*R).toFixed(2)+
  (done?'<br>no mistakes left: every point is on its own side.':'');
}
function step(){
 for(var j=0;j<P.length;j++){
  var i=(idx+j)%P.length,p=P[i];
  if(p[2]*(w[0]*p[0]+w[1]*p[1])<=0){
   hit=i;w[0]+=p[2]*p[0];w[1]+=p[2]*p[1];k++;idx=i+1;done=false;render();return;
  }
 }
 hit=null;done=true;render();
}
function reset(){w=[0,0];k=0;idx=0;hit=null;done=false;render();}
reset();
