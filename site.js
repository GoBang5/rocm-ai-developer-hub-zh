const search=document.querySelector('#search');
const links=[...document.querySelectorAll('a[data-title]')];
function runSearch(){const q=search.value.toLowerCase().trim();let matches=0;for(const a of links){const visible=!q||a.dataset.title.toLowerCase().includes(q);a.hidden=!visible;if(visible)matches++;}for(const d of document.querySelectorAll('.sidebar details')){d.hidden=!!q&&![...d.querySelectorAll('a')].some(a=>!a.hidden);if(q)d.open=true;}document.querySelector('#search-empty').hidden=matches>0;}
search.addEventListener('input',runSearch);
for(const a of links){if(new URL(a.href).pathname===location.pathname){a.classList.add('active');a.setAttribute('aria-current','page');}}
const menu=document.querySelector('#menu'),side=document.querySelector('.sidebar'),shade=document.querySelector('#shade');
function toggleMenu(force){const open=force??!side.classList.contains('open');side.classList.toggle('open',open);shade.classList.toggle('open',open);menu.setAttribute('aria-expanded',String(open));}
menu.addEventListener('click',()=>toggleMenu());shade.addEventListener('click',()=>toggleMenu(false));
document.addEventListener('keydown',e=>{if(e.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){e.preventDefault();if(innerWidth<850)toggleMenu(true);search.focus();}if(e.key==='Escape'){search.value='';runSearch();toggleMenu(false);}});
document.querySelector('#theme').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('rocm-theme',theme);}catch(e){}});
const cards=[...document.querySelectorAll('.tutorial-card')];
function filterCards(){const level=document.querySelector('#level').value,category=document.querySelector('#category').value;let count=0;for(const card of cards){card.hidden=!((level==='all'||card.dataset.level===level)&&(category==='all'||card.dataset.category===category));if(!card.hidden)count++;}document.querySelector('#filter-count').textContent=`找到 ${count} 篇教程`;}
if(cards.length){document.querySelector('#level').addEventListener('change',filterCards);document.querySelector('#category').addEventListener('change',filterCards);filterCards();}
