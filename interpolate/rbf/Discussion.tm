<TeXmacs|2.1.1>

<style|generic>

<\body>
  Given a set of <strong|centers> <math|\<b-x\><rsub|1><rsup|c>,\<ldots\>,\<b-x\><rsub|N><rsup|c>\<in\>\<bbb-R\><rsup|d>>,
  we define the <strong|RBF interpolant>,

  <\equation>
    s<around*|(|\<b-x\>|)>=<big|sum><rsub|j=1><rsup|N>\<alpha\><rsub|j>\<cdot\>\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\>-\<b-x\><rsub|j><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>,<label|rbfInterpolantEq>
  </equation>

  where <math|\<phi\><around*|(|r,\<varepsilon\>|)>=<sqrt|1+\<varepsilon\><rsup|2>r<rsup|2>>>
  is the multiquadric RBF function (MQ). The coefficients
  <math|\<alpha\><rsub|j>> are chosen by enforcing the <strong|interpolation
  condition>: if we want to interpolate a set of points
  <math|<around*|{|\<b-x\><rsub|i>,f<around*|(|\<b-x\><rsub|i>|)>|}>> we
  impose

  <\equation*>
    s<around*|(|\<b-x\><rsub|i>|)>=f<around*|(|\<b-x\><rsub|i>|)>
  </equation*>

  for each <math|\<b-x\><rsub|i>> that will typically be a center. For
  <math|i=1,\<ldots\>,N>, we have the set of conditions

  <\equation>
    <choice|<tformat|<table|<row|<cell|s<around*|(|\<b-x\><rsub|1>|)>=f<around*|(|\<b-x\><rsub|1>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|s<around*|(|\<b-x\><rsub|N>|)>=f<around*|(|\<b-x\><rsub|N>|)>>>>>>.<label|setOfConditions>
  </equation>

  But <eqref|rbfInterpolantEq> can be rewritten as the scalar product

  <\equation*>
    s<around*|(|\<b-x\>|)>=<matrix|<tformat|<table|<row|<cell|\<alpha\><rsub|1>>|<cell|\<alpha\><rsub|2>>|<cell|\<ldots\>>|<cell|\<alpha\><rsub|N>>>>>><matrix|<tformat|<table|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\>-\<b-x\><rsub|1><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\>-\<b-x\><rsub|2><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\>-\<b-x\><rsub|N><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>>>>,
  </equation*>

  hence <eqref|setOfConditions> is rewritten into the system of equations

  <\equation*>
    <wide*|<matrix|<tformat|<table|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|1>-\<b-x\><rsub|1><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|1>-\<b-x\><rsub|2><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<ldots\>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|1>-\<b-x\><rsub|N><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|2>-\<b-x\><rsub|1><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|2>-\<b-x\><rsub|2><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<ldots\>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|2>-\<b-x\><rsub|N><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|>|<cell|\<vdots\>>>|<row|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|N>-\<b-x\><rsub|1><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|N>-\<b-x\><rsub|2><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>|<cell|\<ldots\>>|<cell|\<phi\><around*|(|<around*|\<\|\|\>|\<b-x\><rsub|N>-\<b-x\><rsub|N><rsup|c>|\<\|\|\>><rsub|2>,\<varepsilon\>|)>>>>>>|\<wide-underbrace\>><rsub|<text|interpolation
    matrix>><matrix|<tformat|<table|<row|<cell|\<alpha\><rsub|1>>>|<row|<cell|\<alpha\><rsub|2>>>|<row|<cell|\<vdots\>>>|<row|<cell|\<alpha\><rsub|N>>>>>>=<matrix|<tformat|<table|<row|<cell|f<rsub|1>>>|<row|<cell|f<rsub|2>>>|<row|<cell|\<vdots\>>>|<row|<cell|f<rsub|N>>>>>>.
  </equation*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|rbfInterpolantEq|<tuple|1|1|../../../../../home/heqro/.TeXmacs/texts/scratch/no_name_68.tm>>
    <associate|setOfConditions|<tuple|2|1|../../../../../home/heqro/.TeXmacs/texts/scratch/no_name_68.tm>>
  </collection>
</references>