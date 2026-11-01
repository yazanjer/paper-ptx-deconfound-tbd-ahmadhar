import re
src=open('$HOME/ptx-deconfound/paper/main.tex').read()
pre_end=src.index('\\journal{')
preamble=src[src.index('\\usepackage[T1]'):pre_end].replace('\\usepackage{tikz}\n','')
# drop packages that clash with ieeeaccess
for p in ['\\usepackage{mathptmx}\n','\\usepackage[T1]{fontenc}\n']: preamble=preamble.replace(p,'')
body=src[src.index('%% ============================================================ INTRODUCTION'):src.index('\\section*{Data and code availability}')]
names={'bandaru2026untreated':'Bandaru et al.','feng2022triage':'Feng et al.','graf2020tube':'Graf et al.','jimenez2023shortcuts':'Jim\\\'enez-S\\\'anchez et al.','li2025featurequality':'Li et al.','rueckel2020tubes':'Rueckel et al.','rueckel2021annotations':'Rueckel et al.','seah2021hidden':'Seah et al.','sourget2025clip':'Sourget et al.','thian2021external':'Thian et al.','wollek2023attention':'Wollek et al.'}
body=re.sub(r'\\citet\{([^}]*)\}',lambda m:names[m.group(1)]+'~\\cite{'+m.group(1)+'}',body)
body=body.replace('\\citep{','\\cite{')
body=re.sub(r'\\begin\{(figure|table)\}\[!tbp\]',r'\\begin{\1*}[!t]',body)
body=re.sub(r'\\end\{(figure|table)\}',r'\\end{\1*}',body)
body=body.replace('\\begin{algorithm}[!tbp]','\\begin{algorithm}[!t]')
back=r'''
\section*{Data and code availability}
DATAAVAIL

\section*{Deviations from the pre-registration}
DEVIATIONS

\section*{Ethics statement}
This study used only publicly available, de-identified datasets (SIIM-ACR, NIH ChestX-ray14 and the derived device cohort, PTX-498) under their respective licences. No new patient data were collected; ethical approval was therefore not required. The authors declare no competing interests.

\section*{Acknowledgment}
During the preparation of this work the authors used Claude (Anthropic) to assist with code development, experiment orchestration, statistical analysis, literature search and manuscript drafting. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
'''
da=re.search(r'\\section\*\{Data and code availability\}\n(.*?)\n\n',src,re.S).group(1)
dev=re.search(r'\\section\*\{Deviations from the pre-registration\}\n(.*?\\end\{enumerate\})',src,re.S).group(1)
back=back.replace('DATAAVAIL',da).replace('DEVIATIONS',dev)
abstract=open('$HOME/work/abstract_v0.txt').read().strip()
head=r'''\let\primyear\year
\documentclass{ieeeaccess}
\let\accyear\year\let\year\primyear
\usepackage{tikz}
\let\year\accyear
\definecolor{accessblue}{cmyk}{1,.43,0,.09}% PANTONE 3015C process equivalent (spot colour breaks with pgf)
\usepackage{mciteplus}
\def\xfigwd{0pt}
'''+preamble+r'''
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2026.DOI}

\title{Adversarial Deconfounding With Pseudo Device Labels Does Not Remove the Device Shortcut in Pneumothorax Detection: A Pre-Registered, Multi-Seed External Evaluation of Twelve Models}
\author{\uppercase{Mohanad Dawood Salman}\authorrefmark{1}, \uppercase{Luay Ibrahim Khalaf}\authorrefmark{1}, \uppercase{Omar Ahmed Razooqi}\authorrefmark{2}, and \uppercase{Salwa Khalid Abdulateef}\authorrefmark{1}}
\address[1]{Computer Science Department, College of Computer Science and Mathematics, Tikrit University, Tikrit, Salah al-Deen 34001, Iraq}
\address[2]{Department of Hadith and Its Sciences, College of Islamic Sciences, Tikrit University, Tikrit, Salah al-Deen 34001, Iraq}
\tfootnote{This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.}

\markboth
{Salman \headeretal: Adversarial Deconfounding Does Not Remove the Device Shortcut in Pneumothorax Detection}
{Salman \headeretal: Adversarial Deconfounding Does Not Remove the Device Shortcut in Pneumothorax Detection}

\corresp{Corresponding author: Salwa Khalid Abdulateef (e-mail: khalid.salwa@tu.edu.iq).}

\begin{abstract}
'''+abstract+r'''
\end{abstract}

\begin{keywords}
Chest radiography, deep learning, domain-adversarial training, external validation, medical image segmentation, negative result, pneumothorax, pre-registration, shortcut learning.
\end{keywords}

\titlepgskip=-21pt

\maketitle

'''
bios=r'''
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{salman.png}}]{Mohanad Dawood Salman} received the B.Sc. degree in computer science from Shatt Al-Arab University College, Basra, Iraq, in 2001, and the M.Sc. degree in information technology from Universiti Utara Malaysia (UUM), Malaysia, in 2017. He is currently an Assistant Lecturer with the Department of Computer Science, College of Computer Science and Mathematics, Tikrit University, Iraq. His research interests include artificial intelligence, machine learning and compilers.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{khalaf.png}}]{Luay Ibrahim Khalaf} received the B.Sc. degree in computer science from Al-Mansour College, University of Baghdad, Baghdad, Iraq, in 2000, and the M.Sc. degree in computer science (information systems) from Osmania University, Hyderabad, India, in 2010. He is currently an Assistant Lecturer with the Department of Computer Science, College of Computer Science and Mathematics, Tikrit University, Iraq, where he has taught compilers, software engineering and computer architecture since 2018. His research interests include artificial intelligence, cyber security, data security and computational intelligence.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{razooqi.png}}]{Omar Ahmed Razooqi} received the B.Sc. degree from the College of Computer Science and Mathematics, University of Tikrit, Iraq, in 2010, and the M.Sc. degree in computer science (information technology) from Altinbas University, Turkey, in 2023. He is currently an Assistant Lecturer with the Department of Hadith and Its Sciences, College of Islamic Sciences, Tikrit University, Iraq. His research interests include artificial intelligence, cyber security, data mining, information technology and image processing.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{abdulateef.png}}]{Salwa Khalid Abdulateef} received the B.Sc. and M.Sc. degrees in computer science from the University of Basrah, Iraq. She is currently a Lecturer with the Department of Computer Science, Tikrit University, Iraq. Her research interests include artificial intelligence, optimization, computer vision, image processing, information hiding and machine learning. She can be contacted at khalid.salwa@tu.edu.iq.
\end{IEEEbiography}

\EOD
\end{document}
'''
back=back.replace('\\citep{','\\cite{')
out=head+body+back+'\n\\bibliographystyle{IEEEtranM}\n\\bibliography{refs}\n'+bios
out=out.replace('{figures/','{figures/')
open('$HOME/ptx-deconfound/paper_ieee/main.tex','w').write(out)
print(len(out))
