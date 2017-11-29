# The classifier is a function
X is the independent variable (the features)
y is the dependent variable (the label)

The function is **f(x) = y**. So **f(features) = label**, which you can translate in *"Depending on the features we get the label"*.<br /> 
And **y = mx + b**. This gives the position of the line. We change the parameters m and b to change the line position on the graph. More on this below

In the picture below there is a line drawn between some dots. We can say the dots above the line have one label, and the dots below the line have another label. And the line distinguishes between them.

![Picture from yt channel statisticsfun](Notes Pictures/maxresdefault.jpg?raw=true "Optional Title")

# More examples

When we train our classifier, we give it training data. Let's say we've just give it the first example, some features and their label ( **f(x) = y** is **f(features) = label** ). Now we have a red dot(the first label) on the graph. The line(**mx + b**) is also initialised with random **m** and **b** parameters. At this point we only have one label, so the line doesn't change its position because there is no other label to distinguish from.

![State 1 of the graph](Notes Pictures/firstgraph.png?raw=true "Optional Title")

Now let's add another example to our classifier. A new set of features and this time with a different label(a blue dot) than before. Since we have a different label, the line needs to change the **m** and **b** parameters to change its position between the 2 labels (here we changed the **b** parameter more. This changes the **position** of the line, while **m** changes the **rotation** <br />of the line).<br />
If we would have given it an example with the same label as the first one (another red dot in this case), the line would not have changed, because again, there would be nothing to distinguish from yet.<br />
(In the picture below, the slightly transparent line is to show the *previous position* of the line and how it moved)

![State 2 of the graph](Notes Pictures/secondgraph.png?raw=true "Optional Title")

Now let's add another blue dot. As you can see below, this time the line changed its **rotation** more, so the **m** parameter changed more.

![State 3 of the graph](Notes Pictures/thirdgraph.png?raw=true "Optional Title")

So this is **linear regression**. This is pretty much how the classifier is trained. The more examples we give it, the more it will adjust the line to separate the labels. Then, when we give it test data (so only features). You can imagine a new dot (red or blue) on the graph, and depending on which side of the line it is, the classifier will label it.

![State 4 of the graph](Notes Pictures/fourthgraph.png?raw=true "Optional Title")

The slightly transparent dots represent the test data, so the features we're passing into the classifier to predict if it is a red dot or a blue dot. The classifier will say *"Alright, to my left there are red dots so whatever new is on my left I'll label it 'red dot', and whatever is on my right 'blue dot'"*
