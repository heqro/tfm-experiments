<TeXmacs|2.1.1>

<style|generic>

<\body>
  Given a set of points <math|<around*|{|x<rsub|i>,f<around*|(|x<rsub|i>|)>|}><rsub|i=1><rsup|N>\<subset\>\<bbb-R\><rsup|2>>,
  we want to find a polynomial,\ 

  <\equation*>
    p<rsub|n><around*|(|x|)>=<big|sum><rsub|i=1><rsup|n>a<rsub|i>x<rsup|i>
  </equation*>

  \ that satisfies the <strong|interpolation condition>, namely

  <\equation*>
    <tabular*|<tformat|<table|<row|<cell|p<rsub|n><around*|(|x<rsub|i>|)>=f<around*|(|x<rsub|i>|)>,<space|1em>i=1,\<ldots\>,N.>>|<row|<cell|\<Longupdownarrow\>>>|<row|<cell|<matrix|<tformat|<table|<row|<cell|a<rsub|0>>|<cell|a<rsub|1>>|<cell|\<ldots\>>|<cell|a<rsub|n>>>>>><matrix|<tformat|<table|<row|<cell|1>>|<row|<cell|x<rsub|i>>>|<row|<cell|\<vdots\>>>|<row|<cell|x<rsup|n><rsub|i>>>>>>=f<around*|(|x<rsub|i>|)>,<space|1em>i=1,\<ldots\>,N.>>|<row|<cell|\<Longupdownarrow\>>>|<row|<cell|<wide*|<matrix|<tformat|<table|<row|<cell|1>|<cell|x<rsub|1>>|<cell|x<rsub|1><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|1><rsup|n>>>|<row|<cell|1>|<cell|x<rsub|2>>|<cell|x<rsub|2><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|2><rsup|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|>|<cell|>>|<row|<cell|1>|<cell|x<rsub|N>>|<cell|x<rsub|N><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|N><rsup|n>>>>>>|\<wide-underbrace\>><rsub|N\<times\><around*|(|n+1|)>><wide*|<matrix|<tformat|<table|<row|<cell|a<rsub|0>>>|<row|<cell|a<rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsub|n>>>>>>|\<wide-underbrace\>><rsub|<around*|(|n+1|)>\<times\>1>=<wide*|<matrix|<tformat|<table|<row|<cell|f<around*|(|x<rsub|1>|)>>>|<row|<cell|f<around*|(|x<rsub|2>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|f<around*|(|x<rsub|N>|)>>>>>>|\<wide-underbrace\>><rsub|N\<times\>1>>>>>>
  </equation*>

  Assuming <math|x<rsub|i>\<neq\>x<rsub|j>>, the solution is unique
  <math|\<Leftrightarrow\>> <math|n+1=N>, i.e., there is a (unique)
  polynomial of degree <math|N-1> that passes through each
  <math|<around*|(|x<rsub|i>,f<around*|(|x<rsub|i>|)>|)>>.

  If <math|N\<gtr\>n+1>, note that\ 

  <\equation*>
    <matrix|<tformat|<table|<row|<cell|1>|<cell|x<rsub|1>>|<cell|x<rsub|1><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|1><rsup|n>>>|<row|<cell|1>|<cell|x<rsub|2>>|<cell|x<rsub|2><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|2><rsup|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|>|<cell|>>|<row|<cell|1>|<cell|x<rsub|N>>|<cell|x<rsub|N><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|N><rsup|n>>>>>><rsup|\<top\>><matrix|<tformat|<table|<row|<cell|1>|<cell|x<rsub|1>>|<cell|x<rsub|1><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|1><rsup|n>>>|<row|<cell|1>|<cell|x<rsub|2>>|<cell|x<rsub|2><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|2><rsup|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|>|<cell|>>|<row|<cell|1>|<cell|x<rsub|N>>|<cell|x<rsub|N><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|N><rsup|n>>>>>><matrix|<tformat|<table|<row|<cell|a<rsub|0>>>|<row|<cell|a<rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsub|n>>>>>>=<matrix|<tformat|<table|<row|<cell|1>|<cell|x<rsub|1>>|<cell|x<rsub|1><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|1><rsup|n>>>|<row|<cell|1>|<cell|x<rsub|2>>|<cell|x<rsub|2><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|2><rsup|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|>|<cell|>>|<row|<cell|1>|<cell|x<rsub|N>>|<cell|x<rsub|N><rsup|2>>|<cell|\<ldots\>>|<cell|x<rsub|N><rsup|n>>>>>><rsup|\<top\>><matrix|<tformat|<table|<row|<cell|f<around*|(|x<rsub|1>|)>>>|<row|<cell|f<around*|(|x<rsub|2>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|f<around*|(|x<rsub|N>|)>>>>>>
  </equation*>

  allows us to come upon a system of linear equations <math|A x=b> with
  <math|A\<in\>\<bbb-R\><rsup|<around*|(|n+1|)>,<around*|(|n+1|)>>,b\<in\>\<bbb-R\><rsup|n+1>>.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>