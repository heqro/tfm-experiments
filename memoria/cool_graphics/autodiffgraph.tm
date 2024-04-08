<TeXmacs|2.1.2>

<style|generic>

<\body>
  <with|gr-mode|<tuple|edit|spline>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.5gh>>|gr-geometry|<tuple|geometry|1par|0.6par>|gr-grid|<tuple|empty>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-arrow-end|\|\<gtr\>|gr-grid-old|<tuple|cartesian|<point|0|0>|2>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|<graphics||<\document-at>
    loss (<math|\<cal-L\><around*|(|\<theta\>|)>>)
  </document-at|<point|-0.8999999999999998|-1.5>>|<\document-at>
    parameters <math|<around*|(|\<theta\>|)>>
  </document-at|<point|-6.4|0.6>>|<with|fill-color|pastel
  green|<cline|<point|4.8|1.8>|<point|-3.4|1.8>|<point|-3.4|0.4>|<point|4.8|0.4>>>|<with|fill-color|white|<carc|<point|-2.695834505507758|1.3617522543501215>|<point|-2.0767709296379353|1.1596117913259114>|<point|-2.4136045055077577|0.673112254350121>>>|<with|fill-color|white|<carc|<point|-1.8427760682629999|1.336355296716968>|<point|-1.2237124923931726|1.1342148336927584>|<point|-1.560546068262997|0.647715296716968>>>|<with|fill-color|white|<carc|<point|-0.44277606826299687|1.336355296716968>|<point|0.17628750760682746|1.1342148336927584>|<point|-0.16054606826299572|0.647715296716968>>>|<with|fill-color|white|<carc|<point|0.44387255777422185|1.3588347853915819>|<point|1.0629361336440468|1.1566943223673718>|<point|0.726102557774225|0.6701947853915812>>>|<with|fill-color|white|<carc|<point|1.3359692794490394|1.3597427730586338>|<point|1.955032855318863|1.157602310034425>|<point|1.6181992794490396|0.6711027730586335>>>|<with|fill-color|white|<carc|<point|2.2954942004387497|1.3724209067596815>|<point|2.9145577763085724|1.1702804437354715>|<point|2.5777242004387495|0.6837809067596812>>>|<with|fill-color|white|<carc|<point|3.795494200438747|1.3724209067596824>|<point|4.414557776308573|1.1702804437354715>|<point|4.077724200438749|0.6837809067596812>>>|<math-at|<tabular*|<tformat|<table|<row|<cell|\<partial\>x<rsub|1><separating-space|0.2em>>|<cell|\<partial\>x<rsub|2>>|<cell|\<ldots\>>|<cell|\<partial\>x<rsub|d>>|<cell|<separating-space|0.2em><separating-space|0.2em><separating-space|0.2em>\<partial\>t>|<cell|<separating-space|0.2em><separating-space|0.2em><separating-space|0.2em>\<partial\><rsup|2>x<rsub|1>>|<cell|\<partial\><rsup|2>x<rsub|2>>|<cell|\<ldots\>>|<cell|\<partial\><rsup|k>t>>>>>|<point|-2.8436260682629957|0.9713057282709364>>|<\document-at>
    prediction (<math|\<cal-N\>\<cal-N\><around*|(|\<b-z\>;\<theta\>|)>>)
  </document-at|<point|-1.4|3.0>>|<cline|<point|-1.51756|3.11756>|<point|-1.517561846805133|2.529633549411298>|<point|2.106264056092076|2.529633549411298>|<point|2.106264056092076|3.1175585394893504>>|<\document-at>
    model (<math|\<cal-N\>\<cal-N\>>)
  </document-at|<point|-4.7|3.0>>|<cline|<point|-4.81757|3.11756>|<point|-2.629051461833576|3.1175585394893504>|<point|-2.629051461833576|2.529666622569123>|<point|-4.817568461436698|2.529666622569123>>|<\document-at>
    <tabular*|<tformat|<table|<row|<cell|input (<math|\<b-z\>>)>>>>>
  </document-at|<point|-7.0|3.0>>|<cline|<point|-7.0|3.1>|<point|-5.4|3.11755853948935>|<point|-5.4|2.459121576928166>|<point|-7.0|2.459121576928166>>|<cspline|<point|-6.4|0.1>|<point|-6.4|0.717555232173568>|<point|-4.2|0.717555232173568>|<point|-4.2|0.12963024209551527>>|<with|arrow-end|\|\<gtr\>|<line|<point|-5.39005|2.81535>|<point|-4.817568461436698|2.8236043127397803>>>|<with|arrow-end|\|\<gtr\>|<line|<point|-2.62905|2.8236>|<point|-1.517561846805133|2.823587776160868>>>|<with|arrow-end|\|\<gtr\>|<line|<point|-5.2|0.817209>|<point|-3.723309961635137|2.529666622569123>>>|<\document-at>
    target (<math|P>)
  </document-at|<point|3.1528872866781326|-1.1792499007805264>>|<cline|<point|2.9999951541209153|-1.0999997687524805>|<point|4.824298849054109|-1.1>|<point|4.824298849054109|-1.6878919169202282>|<point|3.0|-1.6878919169202282>>|<cline|<point|-1.01756|-1.38244>|<point|0.854990739515809|-1.3824414605106496>|<point|0.854990739515809|-1.9703664505887022>|<point|-1.017561846805133|-1.9703664505887022>>|<\document-at>
    optimizer
  </document-at|<point|-5|-1>>|<with|arrow-end|\|\<gtr\>|<line|<point|0.294351|2.52963>|<point|0.2999999999999996|1.8>>>|<cline|<point|-5.11756|-0.882441>|<point|-3.431571636459849|-0.8824414605106495>|<point|-3.431571636459849|-1.470333377430877>|<point|-5.117558539489351|-1.470333377430877>>|<spline|<point|4.192248703532212|2.223944741367906>|<point|4.892248703532214|1.9239447413679058>|<point|4.892248703532214|0.3239447413679059>|<point|4.192248703532212|-0.0760552586320942>>|<\document-at>
    <tabular*|<tformat|<table|<row|<cell|automatic>>|<row|<cell|differentiation>>>>>
  </document-at|<point|5.064079243286149|1.496024606429422>>|<with|arrow-end|\|\<gtr\>|<line|<point|0.4|0.4>|<point|-0.08129382193411827|-1.3824414605106496>>>|<with|arrow-end|\|\<gtr\>|<line|<point|-1.01756|-1.67641>|<point|-3.431571636459849|-1.1763956872602197>>>|<with|arrow-end|\|\<gtr\>|<spline|<point|-5.11756|-1.1764>|<point|-5.5|-0.6>|<point|-5.5|0.0>>>|<with|arrow-end|\|\<gtr\>|<spline|<point|3|-1.4>|<point|2.0|-1.7>|<point|0.854990739515809|-1.6764122238391321>>>>>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>