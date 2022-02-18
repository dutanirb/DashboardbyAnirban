class Anirban_Dashboard():
    def __init__(self, train, c, y, manual=True):
        self.train = train
        self.c = c
        self.y = y
        self.X = self.train.drop(self.y,1)
        self.manual = manual

    def make_charts(self):
        import pandas as pd
        import ipywidgets as widgets
        import plotly.express as px
        import plotly.figure_factory as ff
        import plotly.offline as pyo
        from ipywidgets import HBox, VBox, Button
        from ipywidgets import interact, interact_manual, interactive
        import plotly.graph_objects as go
        from plotly.offline import iplot

        header = widgets.HTML(value="<h2>Welcome to Anirban's Dashboard </h2>")
        display(header)


        if len(self.train) > 500:
            from sklearn.model_selection import train_test_split
            test_size = 500/len(self.train)
            if self.c!=None:
                data = self.X.drop(self.c,1)
            else:
                data = self.X

            target = self.train[self.y]
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=1)
            train_mc = pd.concat([X_test, y_test], axis=1)
        else:
            train_mc = self.train

        train_numeric = train_mc.select_dtypes('number')
        train_cat = train_mc.select_dtypes(exclude='number')

        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8, out9])
        tab.set_title(0, 'Scatter Plot')
        tab.set_title(1, 'Pie Chart')
        tab.set_title(2, 'Bar Plot')
        tab.set_title(3, 'Violin Plot')
        tab.set_title(4, 'Box Plot')
        tab.set_title(5, 'Distribution Plot')
        tab.set_title(6, 'Histogram')
        tab.set_title(7, 'Correlation plot')
        tab.set_title(8, 'Dashboard Summary')

        display(tab)

        with out1:
            header = widgets.HTML(value="<h1>Scatter Plots </h1>")
            display(header)

            x = widgets.Dropdown(options=list(train_mc.select_dtypes('number').columns))
            def scatter_plot(X_Axis=list(train_mc.select_dtypes('number').columns),
                            Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                            Color=list(train_mc.select_dtypes('number').columns)):

                fig = go.FigureWidget(data=go.Scatter(x=train_mc[X_Axis],
                                                y=train_mc[Y_Axis],
                                                mode='markers',
                                                text=list(train_cat),
                                                marker_color=train_mc[Color]))

                fig.update_layout(title=f'{Y_Axis.title()} vs {X_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig.show()

            widgets.interact_manual.opts['manual_name'] = 'Make_Chart'
            one = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)

        with out2:
            header = widgets.HTML(value="<h1>Pie Charts </h1>")
            display(header)

            def pie_chart(Labels=list(train_mc.select_dtypes(exclude='number').columns),
                        Values=list(train_mc.select_dtypes('number').columns)[0:]):

                fig = go.FigureWidget(data=[go.Pie(labels=train_mc[Labels], values=train_mc[Values])])

                fig.update_layout(title=f'{Values.title()} vs {Labels.title()}',
                                autosize=False,width=500,height=500)
                fig.show()
            one = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)

        with out3:
            header = widgets.HTML(value="<h1>Bar Plots </h1>")
            display(header)

            def bar_plot(X_Axis=list(train_mc.select_dtypes(exclude='number').columns),
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig1 = px.bar(train_mc, x=train_mc[X_Axis], y=train_mc[Y_Axis], color=train_mc[Color])
                fig1.update_layout(barmode='group',
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig1.show()
            one = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)


        with out4:
            header = widgets.HTML(value="<h1>Violin Plots </h1>")
            display(header)

            def viol_plot(X_Axis=list(train_mc.select_dtypes('number').columns),
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig2 = px.violin(train_mc, X_Axis, Y_Axis, Color, box=True, hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()

            one = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)


        with out5:
            header = widgets.HTML(value="<h1>Box Plots </h1>")
            display(header)

            def box_plot(X_Axis=list(train_mc.select_dtypes(exclude='number').columns),
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[0:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):


                fig4 = px.box(train_mc, x=X_Axis, y=Y_Axis, color=Color, points="all")

                fig4.update_layout(barmode='group',
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig4.show()

            one = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)

        with out6:
            header = widgets.HTML(value="<h1>Distribution Plots </h1>")
            display(header)

            def dist_plot(X_Axis=list(train_mc.select_dtypes('number').columns),
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig2 = px.histogram(train_mc, X_Axis, Y_Axis, Color, marginal='violin', hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()

            one = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)

        with out7:
            header = widgets.HTML(value="<h1>Histogram </h1>")
            display(header)

            def hist_plot(X_Axis=list(train_mc.columns)):
                fig2 = px.histogram(train_mc, X_Axis)
                fig2.update_layout(title=f'{X_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()


            one = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            h = widgets.HBox([three, four])
            i = widgets.VBox([g,h])
            display(i)

        with out8:
            header = widgets.HTML(value="<h1>Correlation Plots </h1>")
            display(header)
            import plotly.figure_factory as ff
            corrs = train_mc.corr()
            colorscale = ['Greys', 'Greens', 'Bluered', 'RdBu',
                    'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
                    'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
            @interact_manual
            def plot_corrs(colorscale=colorscale):
                figure = ff.create_annotated_heatmap(z = corrs.round(2).values,
                                                x =list(corrs.columns),
                                                y=list(corrs.index),
                                                colorscale=colorscale,
                                                annotation_text=corrs.round(2).values)
                iplot(figure)

        with out9:
            header = widgets.HTML(value="<h1> Dashboard Summary </h1>")
            display(header)
            x = widgets.Dropdown(options=list(train_mc.select_dtypes('number').columns))

            def dashboard(X_Axis=list(train_mc.select_dtypes('number').columns),
                             Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                             Color=list(train_mc.select_dtypes('number').columns)):
                fig = go.FigureWidget(data=go.Scatter(x=train_mc[X_Axis],
                                                      y=train_mc[Y_Axis],
                                                      mode='markers',
                                                      text=list(train_cat),
                                                      marker_color=train_mc[Color]))

                fig.update_layout(title=f'{Y_Axis.title()} vs {X_Axis.title()}',
                                  xaxis_title=f'{X_Axis.title()}',
                                  yaxis_title=f'{Y_Axis.title()}',
                                  autosize=False, width=600, height=600)
                fig.show()

            widgets.interact_manual.opts['manual_name'] = 'Make_Chart'
            one = interactive(dashboard, {'manual': self.manual, 'manual_name': 'Make_Chart'})
            #two = interactive(scatter_plot, {'manual': self.manual, 'manual_name': 'Make_Chart'})
            #three = interactive(scatter_plot, {'manual': self.manual, 'manual_name': 'Make_Chart'})
            #four = interactive(scatter_plot, {'manual': self.manual, 'manual_name': 'Make_Chart'})

            g = widgets.HBox([one])
            #h = widgets.HBox([three,four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Scatter Plot </h6>")
            display(header)
            display(i)
            def pie_chart2(Labels=list(train_mc.select_dtypes(exclude='number').columns),
                        Values=list(train_mc.select_dtypes('number').columns)[0:]):

                fig = go.FigureWidget(data=[go.Pie(labels=train_mc[Labels], values=train_mc[Values])])

                fig.update_layout(title=f'{Values.title()} vs {Labels.title()}',
                                autosize=False,width=500,height=500)
                fig.show()
            one = interactive(pie_chart2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #two = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one])
            #h = widgets.HBox([three])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Pie Chart </h6>")
            display(header)
            display(i)
            def bar_plot2(X_Axis=list(train_mc.select_dtypes(exclude='number').columns),
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig1 = px.bar(train_mc, x=train_mc[X_Axis], y=train_mc[Y_Axis], color=train_mc[Color])
                fig1.update_layout(barmode='group',
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig1.show()
            #one = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(bar_plot2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([two])
            #h = widgets.HBox([three, four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Bar Plot </h6>")
            display(header)
            display(i)
            def viol_plot2(X_Axis=list(train_mc.select_dtypes('number').columns),
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig2 = px.violin(train_mc, X_Axis, Y_Axis, Color, box=True, hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()

            one = interactive(viol_plot2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #two = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one])
            #h = widgets.HBox([three, four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Violin Plot </h6>")
            display(header)
            display(i)
            def box_plot2(X_Axis=list(train_mc.select_dtypes(exclude='number').columns),
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[0:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):


                fig4 = px.box(train_mc, x=X_Axis, y=Y_Axis, color=Color, points="all")

                fig4.update_layout(barmode='group',
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig4.show()

            one = interactive(box_plot2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #two = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one])
            #h = widgets.HBox([three, four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Box Plot </h6>")
            display(header)
            display(i)
            def dist_plot2(X_Axis=list(train_mc.select_dtypes('number').columns),
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig2 = px.histogram(train_mc, X_Axis, Y_Axis, Color, marginal='violin', hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()

            one = interactive(dist_plot2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #two = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one])
            #h = widgets.HBox([three, four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Distribution Plot </h6>")
            display(header)
            display(i)
            def hist_plot2(X_Axis=list(train_mc.columns)):
                fig2 = px.histogram(train_mc, X_Axis)
                fig2.update_layout(title=f'{X_Axis.title()}',
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()


            one = interactive(hist_plot2, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #two = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #three = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            #four = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            #h = widgets.HBox([three, four])
            i = widgets.VBox([g])
            header = widgets.HTML(value="<h6> Histogram </h6>")
            display(header)
            display(i)



            import plotly.figure_factory as ff
            corrs = train_mc.corr()
            colorscale = ['Greys', 'Greens', 'Bluered', 'RdBu',
                          'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
                          'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']

            @interact_manual
            def plot_corrs(colorscale=colorscale):
                figure = ff.create_annotated_heatmap(z=corrs.round(2).values,
                                                     x=list(corrs.columns),
                                                     y=list(corrs.index),
                                                     colorscale=colorscale,
                                                     annotation_text=corrs.round(2).values)
                header = widgets.HTML(value="<h6> Correlation Plot </h6>")
                display(header)
                iplot(figure)




