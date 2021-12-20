# importações necessárias
import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(df_column, filename):
    value_counts = df_column.value_counts(normalize=True)

    # informações específicas do gráfico
    bar1_values = list(value_counts.values)

    x_axis_labels = value_counts.index.tolist() # legendas dos bars do eixo x
    # y_axis_labels = ['0', '', '20', '', '40', '', '60', '', '80', '']

    n = len(value_counts) # quantidade de bars
    width = 1 # largura do bar

    ind = list(range(1,n*2,2)) # vetor de inteiros de 0 a n-1
    bar1_color = 'lightskyblue' # cor do bar

    figsize = n*4-1 if n <5 else n*2-1
    plt.rcParams['figure.figsize'] = figsize, 9 # redimensionando figura
    fig, ax = plt.subplots() # figura e axes

    # ax.set_title('Distribuição', fontsize=25, pad=30) # título

    ax.grid(color='whitesmoke', linestyle='-', linewidth=2) # grid
    ax.grid(zorder=0) # parâmetro pra colocar
    ax.xaxis.grid(False) # grid vertical

    ax.spines['right'].set_visible(False) # contorno da direita
    ax.spines['top'].set_visible(False) # contorno de cima

    bar1 = ax.bar(ind, bar1_values, width, color=bar1_color, zorder=3) # segundo bar

    ax.set_xlabel('Classes', labelpad=20, fontsize=20) # label do eixo x
    ax.tick_params(labelsize=15, length=0) # tamanho da fonte dos eixos

    ax.set_xticklabels(x_axis_labels, fontsize=15, color='black') # labels do bar no eixo x (forma 1)
    ax.set_xticks(ind) # quantidade de bars no eixo x

    plt.draw()

    ax.set_ylabel('Populacao (%)', labelpad=20, fontsize=20) # label do eixo y
    y_axis_labels = [str(int(float(item.get_text())*100)) if ((i%2==0) and (item.get_text() != '')) else '' for i, item in enumerate(ax.get_yticklabels())]
    ax.set_yticklabels(y_axis_labels, fontsize=15, color='dimgray') # labels do bar no eixo y (forma 2)

    labels_x = ax.get_xticklabels()  # labels do eixo  x
    labels_y = ax.get_yticklabels()  # labels do eixo y


    ax.set_xlim(0, n*2+0.5)

    # espaço entre gráfico e labels do eixo x
    for x in labels_x:
        x.set_y(-0.02)

    # espaço entre gráfico e labels do eixo y
    for y in labels_y:
        y.set_x(-0.005)

    # função para inserir valor do bar acima de cada bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='black', weight=550, clip_on=True, fontsize=13)

    # chamada da função autolabel para cada bar 
    autolabel(bar1)

    ax.autoscale_view() # ajustes de escala
    plt.tight_layout() # ajustes de tamanho do fundo do gráfico
    
    plt.savefig(filename) # salvar gráfico
    plt.show() # mostrar gráfico
    

def plot_both_distributions(params1, params2, filename):
    legend1, df_column1 = params1
    legend2, df_column2 = params2
    
    value_counts1 = df_column1.value_counts(normalize=True)
    value_counts2 = df_column2.value_counts(normalize=True)

    # informações específicas do gráfico
    bar1_values = list(value_counts1.values)
    bar2_values = list(value_counts2.values)

    x_axis_labels = value_counts1.index.tolist() # legendas dos bars do eixo x
    # y_axis_labels = ['0', '', '20', '', '40', '', '60', '', '80', '']

    n = len(value_counts1) # quantidade de bars
    width = 0.4 # largura do bar

    ind = np.arange(n)*2 # vetor de inteiros de 0 a n-1
    bar1_color = 'lightskyblue' # cor do primeiro bar
    bar2_color = 'steelblue' # cor do segundo bar

    plt.rcParams['figure.figsize'] = 7, 9 # redimensionando figura
    fig, ax = plt.subplots() # figura e axes

    # ax.set_title('Distribuição', fontsize=25, pad=30) # título

    ax.grid(color='whitesmoke', linestyle='-', linewidth=2) # grid
    ax.grid(zorder=0) # parâmetro pra colocar
    ax.xaxis.grid(False) # grid vertical

    ax.spines['right'].set_visible(False) # contorno da direita
    ax.spines['top'].set_visible(False) # contorno de cima

    bar1 = ax.bar(ind - width, bar1_values, width, color=bar1_color, zorder=3) # primeiro bar
    bar2 = ax.bar(ind, bar2_values, width, color=bar2_color, zorder=3) # segundo bar

    ax.set_xlabel('Classes', labelpad=20, fontsize=20) # label do eixo x
    ax.tick_params(labelsize=15, length=0) # tamanho da fonte dos eixos

    ax.set_xticklabels(x_axis_labels, fontsize=15, color='black') # labels do bar no eixo x (forma 1)
    ax.set_xticks(ind) # quantidade de bars no eixo x
    
    ax.legend((bar1[0], bar2[0]), 
      (legend1, legend2),
      fontsize=15, edgecolor='white', framealpha=1,
      loc='center right') # legenda do gráfico

    plt.draw()

    ax.set_ylabel('Populacao (%)', labelpad=20, fontsize=20) # label do eixo y
    y_axis_labels = [str(int(float(item.get_text())*100)) if ((i%2==0) and (item.get_text() != '')) else '' for i, item in enumerate(ax.get_yticklabels())]
    ax.set_yticklabels(y_axis_labels, fontsize=15, color='dimgray') # labels do bar no eixo y (forma 2)

    labels_x = ax.get_xticklabels()  # labels do eixo  x
    labels_y = ax.get_yticklabels()  # labels do eixo y


    ax.set_xlim(-1, 3)

    # espaço entre gráfico e labels do eixo x
    for x in labels_x:
        x.set_y(-0.02)

    # espaço entre gráfico e labels do eixo y
    for y in labels_y:
        y.set_x(-0.005)

    # função para inserir valor do bar acima de cada bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='black', weight=550, clip_on=True, fontsize=13)

    # chamada da função autolabel para cada bar 
    autolabel(bar1)
    autolabel(bar2)

    ax.autoscale_view() # ajustes de escala
    plt.tight_layout() # ajustes de tamanho do fundo do gráfico
    
    plt.savefig(filename) # salvar gráfico
    plt.show() # mostrar gráfico