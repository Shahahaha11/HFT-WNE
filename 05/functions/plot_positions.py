# function  plot_positions_ma

# function drawing the 1MA strategy framework
# for a single day

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_ma(data_plot,      # DataFrame with DatetimeIndex
                      date_plot,      # Date as string 'YYYY-MM-DD'
                      col_price,      # name of the column with price
                      col_ma,         # name of the column with MA/median
                      col_pos,        # name of the column with position (-1, 0, 1)
                      title,          # title of te plot
                      save_graph = False,
                      width = 10,
                      height = 6,
                      file_name = None):
    """
    Draws a price chart with MA and positions (colored background).
    """

    # Filter data to a given date
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_price, col_ma, col_pos]]

    # We add the time column and next_time (to the rectangles)
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Colors of positions
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # 1. Initialization of the plot
    fig, ax = plt.subplots(figsize=(width, height))

    # 2. First we draw the lines (price and MA)
    ax.plot(data_day["Time"], data_day[col_price], color='black', label='Price')
    ax.plot(data_day["Time"], data_day[col_ma], color='blue', label='MA', linewidth=1.5)

    # 3. Only now we get the Y-axis limits
    ymin, ymax = ax.get_ylim()

    # 4. We add a colored background for the positions
    for i, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                    row['next_Time'] - row['Time'],
                    ymax - ymin,
                    facecolor=color, alpha=0.2)
        )

    # We draw line graphs
    ax.plot(data_day["Time"], data_day[col_price], color='black', label='Price')
    ax.plot(data_day["Time"], data_day[col_ma], color='blue', label='MA', linewidth=1.5)

    # Formatting of axes
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price / signal")
    # Remove duplicates in the legend (if there are any)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # unique: label → line
    ax.legend(by_label.values(), by_label.keys(), loc='upper left') 
    # ax.grid(True)
    plt.xticks(rotation=0)

    # Formatting of X axis – only time (HH:MM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    # Save the chart if necessary
    if save_graph:
        if file_name is None:
            raise ValueError("You have to provide a file_name if save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax


#-----------------------------------------------
# function plot_positions_2mas

# function drawing the 2MAs strategy framework
# for a single day

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_2mas(data_plot,      # DataFrame with DatetimeIndex
                        date_plot,      # Date as string 'YYYY-MM-DD'
                        col_price,      # name of the column with the price
                        col_fma,        # name of the column with the fast moving average
                        col_sma,        # name of the column with the slow moving average
                        col_pos,        # name of the column with position (-1, 0, 1)
                        title,          # title of the plot
                        save_graph  =False,
                        width = 10,
                        height = 6,
                        file_name = None):
    """
    Draws a chart of price, two averages (fma and sma) and positions as a colored background.
    """

    # Filtering data to a specific date
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_price, col_fma, col_sma, col_pos]].copy()

    # Preparing time columns
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Background colors depending on position
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # Initialization of the plot
    fig, ax = plt.subplots(figsize=(width, height))

    # We draw lines: price, FMA, SMA
    ax.plot(data_day["Time"], data_day[col_price], color='gray', label='Price')
    ax.plot(data_day["Time"], data_day[col_fma], color='blue', label='Fast MA', linewidth=1.5)
    ax.plot(data_day["Time"], data_day[col_sma], color='darkgreen', label='Slow MA', linewidth=1.5)

    # Get the current Y-axis limits after drawing the line
    ymin, ymax = ax.get_ylim()

    # Add rectangles as background for positions
    for _, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                      row['next_Time'] - row['Time'],
                      ymax - ymin,
                      facecolor=color,
                      alpha=0.2)
        )

    # Axis formatting and legend
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price / signal")

    # Remove duplicates in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # X-axis – time format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Ticks every 30 minutes, starting from the first time
    start = data_day["Time"].iloc[0]
    end = data_day["Time"].iloc[-1]
    tick_locs = pd.date_range(start=start, end=end, freq='30min')
    ax.set_xticks(tick_locs)
    ax.set_xlim(start, end)

    # Write to file if required
    if save_graph:
        if file_name is None:
            raise ValueError("You have to provide a file_name if save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax



#-----------------------------------------------------
# function plot_positions_vb

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_vb(data_plot,       # DataFrame with DatetimeIndex
                      date_plot,       # Date as string 'YYYY-MM-DD'
                      col_signal,      # name of the column with the signal
                      col_upper,       # name of the column with the upper boundary
                      col_lower,       # name of the column with the lower boundary
                      col_pos,         # name of the column with position (-1, 0, 1)
                      title,           # title of the plot
                      save_graph = False,
                      width = 10,
                      height = 6,
                      file_name = None):
    """
    Draws a signal graph with a range (upper/lower) and background by position.
    """

    # Filtering data to a specific date
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_signal, col_upper, col_lower, col_pos]].copy()

    # Preparing time columns
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Colors of positions
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # Initialization of the plot
    fig, ax = plt.subplots(figsize=(width, height))

    # Rysujemy sygnał i przedział (dolna i górna linia)
    ax.plot(data_day["Time"], data_day[col_signal], color='black', label='Signal')
    ax.plot(data_day["Time"], data_day[col_upper], color='blue', linewidth=1.5, label='Upper Bound')
    ax.plot(data_day["Time"], data_day[col_lower], color='blue', linewidth=1.5, label='Lower Bound')

    # We set the Y range after drawing the line
    ymin, ymax = ax.get_ylim()

    # Background by position
    for _, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                      row['next_Time'] - row['Time'],
                      ymax - ymin,
                      facecolor=color,
                      alpha=0.2)
        )

    # Axis formatting and legend
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")

    # Removing duplicates in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Formatting X-axis: HH:MM only
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Ticks every 15 minutes from the start of data
    start = data_day["Time"].iloc[0]
    end = data_day["Time"].iloc[-1]
    tick_locs = pd.date_range(start=start, end=end, freq='15min')
    ax.set_xticks(tick_locs)
    ax.set_xlim(start, end)
    plt.xticks(rotation=45)

    # Save the graph to a file if required
    if save_graph:
        if file_name is None:
            raise ValueError("You have to provide a file_name if save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax
   