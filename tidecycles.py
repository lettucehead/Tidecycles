import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.image as mpimg

# # Dataframes
# Lunisolar Daily
# Sessions
# Hilo
# Perigee/Apogee
# Full/New
# North/South
# Equator
# Ascending/Descending?

# # Variables of Interest
# Date
# Julian date
# Ecliptic longitude
# Cycle

daily_jdate = pd.read_csv("1980-2080-daily-jdate.csv", sep = ",", header=9, parse_dates = ["UTC calendar date"])
daily_jdate.drop(daily_jdate.tail(4).index,inplace=True)
daily_jdate["UTC Julian date"] = daily_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
daily_lunar_eclip_longitude = pd.read_csv("1980-2080-daily-lunar-eclip-longitude.csv", sep = ",", header=15, parse_dates = ["UTC calendar date"])
daily_lunar_eclip_longitude.drop(daily_lunar_eclip_longitude.tail(4).index,inplace=True)
daily_lunar_eclip_longitude = pd.merge(daily_jdate, daily_lunar_eclip_longitude)
daily_solar_eclip_longitude = pd.read_csv("1980-2080-daily-solar-eclip-longitude.csv", sep = ",", header=15, parse_dates = ["UTC calendar date"])
daily_solar_eclip_longitude.drop(daily_solar_eclip_longitude.tail(4).index,inplace=True)
daily_solar_eclip_longitude['Solar Ecliptic Longitude'] = daily_solar_eclip_longitude['Longitude (deg)'] % 360
daily_solar_eclip_longitude["UTC calendar date"] = pd.to_datetime(daily_solar_eclip_longitude["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
solar2 = daily_solar_eclip_longitude.drop(columns = ['Longitude (deg)', 'Latitude (deg)', 'Radius (km)', 'd Longitude/dt (deg/s)', 'd Latitude/dt (deg/s)', 'd Radius/dt (km/s)', 'Speed (km/s)','Time at Target','Light Time (s)'])
daily_lunar_eclip_longitude["UTC calendar date"] = pd.to_datetime(daily_lunar_eclip_longitude["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
daily_lunar_eclip_longitude = pd.merge(daily_lunar_eclip_longitude, solar2)
daily_lunar_eclip_longitude['Ecliptic Longitude'] = daily_lunar_eclip_longitude['Longitude (deg)'] % 360
daily_lunar_eclip_longitude['Sidereal Cycles'] = (daily_lunar_eclip_longitude['Ecliptic Longitude'] < daily_lunar_eclip_longitude['Ecliptic Longitude'].shift(1)).cumsum()

# To get ecliptic longitude of sessions, you'll need these CSVs.
# sessions['StartDatetime'].to_csv("starts.csv", index=False)
# sessions['EndDatetime'].to_csv("ends.csv", index=False)

hourly = pd.read_csv("2023-2042-hourly-pred.csv", sep = ",", parse_dates = ["Date Time"])
hourly_jdate = pd.read_csv("jhourly.csv", sep = ",", header=9)
hourly_jdate.drop(hourly_jdate.tail(4).index,inplace=True)
hourly_jdate["UTC Julian date"] = hourly_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
hourly_jdate["UTC calendar date"] = pd.to_datetime(hourly_jdate["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
hourly["UTC calendar date"] = hourly["Date Time"]
jhourly = pd.merge(hourly_jdate, hourly)
hourly_ecl = pd.read_csv("2023-2042-hourly-ecliptic-longitude.csv", sep = ",", header=15, parse_dates = ["UTC calendar date"])
hourly_ecl.drop(hourly_ecl.tail(4).index,inplace=True)
hourly_ecl["UTC calendar date"] = pd.to_datetime(hourly_ecl["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
hourly_ecl = pd.merge(jhourly, hourly_ecl)
hourly_ecl['Ecliptic Longitude'] = hourly_ecl['Longitude (deg)'] % 360
hourly_ecl = hourly_ecl.drop(columns=['Date Time'])

hilo = pd.read_csv("2023-2042-hilo-data.csv", sep = ",", parse_dates = ["Datetime"])
hilo_jdate = pd.read_csv("jhilo.csv", sep = ",", header=8)
hilo_jdate.drop(hilo_jdate.tail(4).index,inplace=True)
hilo_jdate["UTC Julian date"] = hilo_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
hilo_jdate["UTC calendar date"] = pd.to_datetime(hilo_jdate["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
hilo["UTC calendar date"] = hilo["Datetime"]
jhilo = pd.merge(hilo_jdate, hilo)
hilo_ecl = pd.read_csv("2023-2042-hilo-ecliptic-longitude.csv", sep = ",", header=14, parse_dates = ["UTC calendar date"])
hilo_ecl.drop(hilo_ecl.tail(4).index,inplace=True)
hilo_ecl["UTC calendar date"] = pd.to_datetime(hilo_ecl["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
hilo_ecl = pd.merge(jhilo, hilo_ecl)
hilo_ecl['Ecliptic Longitude'] = hilo_ecl['Longitude (deg)'] % 360
hilo_ecl = hilo_ecl.drop(columns=['Datetime'])
hilo_ecl['Cycle'] = hilo_ecl['Ecliptic Longitude'] - hilo_ecl['Ecliptic Longitude'].shift(1) < 0
hilo_ecl['Sidereal Cycles'] = hilo_ecl['Cycle'].cumsum() + 575

sessions = pd.read_csv("2023-2042-sixmin-sessions.csv", sep = ",", parse_dates = ["StartDatetime", "EndDatetime"])
sessions_pred = pd.read_csv("2023-2042-sixmin-sessions-pred.csv", sep = ",", parse_dates = ["StartDatetime", "EndDatetime"])
starts_pred = sessions_pred.loc[:,['StartDatetime','StartPred']]
starts_jdate = pd.read_csv("starts_jdate.csv", sep = ",", header=8, parse_dates = ["UTC calendar date"])
starts_jdate.drop(starts_jdate.tail(4).index,inplace=True)
starts_jdate["UTC Julian date"] = starts_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
starts_jdate["UTC calendar date"] = pd.to_datetime(starts_jdate["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
starts_pred['UTC calendar date'] = starts_pred['StartDatetime']
starts_pred = pd.merge(starts_jdate, starts_pred)
starts_pred['Prediction'] = starts_pred['StartPred']
starts_pred = starts_pred.drop(columns=['StartDatetime', 'StartPred'])
starts_eclip_longitude = pd.read_csv("starts_ecliptic_longitude.csv", sep = ",", header=14, parse_dates = ["UTC calendar date"])
starts_eclip_longitude.drop(starts_eclip_longitude.tail(4).index,inplace=True)
starts_eclip_longitude["UTC calendar date"] = pd.to_datetime(starts_eclip_longitude["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
starts_eclip_longitude = pd.merge(starts_pred, starts_eclip_longitude)
starts_eclip_longitude['Ecliptic Longitude'] = starts_eclip_longitude['Longitude (deg)'] % 360
starts_ecl = pd.merge(starts_eclip_longitude, starts_pred)
ends_pred = sessions_pred.loc[:,['EndDatetime','EndPred']]
ends_jdate = pd.read_csv("ends_jdate.csv", sep = ",", header=8, parse_dates = ["UTC calendar date"])
ends_jdate.drop(ends_jdate.tail(4).index,inplace=True)
ends_jdate["UTC Julian date"] = ends_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
ends_jdate["UTC calendar date"] = pd.to_datetime(ends_jdate["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
ends_pred['UTC calendar date'] = ends_pred['EndDatetime']
ends_pred = pd.merge(ends_jdate, ends_pred)
ends_pred['Prediction'] = ends_pred['EndPred']
ends_pred = ends_pred.drop(columns=['EndDatetime', 'EndPred'])
ends_eclip_longitude = pd.read_csv("ends_ecliptic_longitude.csv", sep = ",", header=14, parse_dates = ["UTC calendar date"])
ends_eclip_longitude.drop(ends_eclip_longitude.tail(4).index,inplace=True)
ends_eclip_longitude["UTC calendar date"] = pd.to_datetime(ends_eclip_longitude["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
ends_eclip_longitude = pd.merge(ends_pred, ends_eclip_longitude)
ends_eclip_longitude['Ecliptic Longitude'] = ends_eclip_longitude['Longitude (deg)'] % 360
ends_ecl = pd.merge(ends_eclip_longitude, ends_pred)

bookends = pd.concat([starts_ecl, ends_ecl], ignore_index=True)
hourly_ecl = pd.concat([hourly_ecl, bookends], ignore_index=True).sort_values('UTC calendar date')
hourly_ecl = pd.concat([hilo_ecl, hourly_ecl], ignore_index=True).sort_values('UTC calendar date')
hourly_ecl = hourly_ecl.reset_index(drop=True)

hourly_ecl['Cycle'] = hourly_ecl['Ecliptic Longitude'] - hourly_ecl['Ecliptic Longitude'].shift(1) < 0
hourly_ecl['Sidereal Cycles'] = hourly_ecl['Cycle'].cumsum() + 575

apogee_jdate = pd.read_csv("1980-2080-apogee-jdate.csv", sep = ",", header=9, parse_dates = ["UTC calendar date"])
apogee_jdate.drop(apogee_jdate.tail(4).index,inplace=True)
apogee_jdate["UTC Julian date"] = apogee_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
apogee = pd.read_csv("1980-2080-apogee.csv", sep = ",", header=15, parse_dates = ["UTC calendar date"])
apogee.drop(apogee.tail(4).index,inplace=True)
apogee = pd.merge(apogee_jdate, apogee)
apogee["UTC calendar date"] = pd.to_datetime(apogee["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
apogee['Ecliptic Longitude'] = apogee['Longitude (deg)'] % 360

perigee_jdate = pd.read_csv("1980-2080-perigee-jdate.csv", sep = ",", header=9)
perigee_jdate.drop(perigee_jdate.tail(4).index,inplace=True)
perigee_jdate["UTC Julian date"] = perigee_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
perigee = pd.read_csv("1980-2080-perigee.csv", sep = ",", header=15)
perigee.drop(perigee.tail(4).index,inplace=True)
perigee = pd.merge(perigee_jdate, perigee)
perigee["UTC calendar date"] = pd.to_datetime(perigee["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
perigee['Ecliptic Longitude'] = perigee['Longitude (deg)'] % 360

fulls_jdate = pd.read_csv("1980-2080-fulls-jdate.csv", sep = ",", header=9)
fulls_jdate.drop(fulls_jdate.tail(4).index,inplace=True)
fulls_jdate["UTC Julian date"] = fulls_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
fulls = pd.read_csv("1980-2080-fulls.csv", sep = ",", header=15)
fulls.drop(fulls.tail(4).index,inplace=True)
fulls = pd.merge(fulls_jdate, fulls)
fulls["UTC calendar date"] = pd.to_datetime(fulls["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
fulls['Ecliptic Longitude'] = fulls['Longitude (deg)'] % 360

news_jdate = pd.read_csv("1980-2080-news-jdate.csv", sep = ",", header=9)
news_jdate.drop(news_jdate.tail(4).index,inplace=True)
news_jdate["UTC Julian date"] = news_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
news = pd.read_csv("1980-2080-news.csv", sep = ",", header=15)
news.drop(news.tail(4).index,inplace=True)
news = pd.merge(news_jdate, news)
news["UTC calendar date"] = pd.to_datetime(news["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
news['Ecliptic Longitude'] = news['Longitude (deg)'] % 360

norths_jdate = pd.read_csv("1980-2080-norths-jdate.csv", sep = ",", header=9)
norths_jdate.drop(norths_jdate.tail(4).index,inplace=True)
norths_jdate["UTC Julian date"] = norths_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
norths = pd.read_csv("1980-2080-norths-ecliptic-longitude.csv", sep = ",", header=15)
norths.drop(norths.tail(4).index,inplace=True)
norths = pd.merge(norths_jdate, norths)
norths["UTC calendar date"] = pd.to_datetime(norths["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
norths['Ecliptic Longitude'] = norths['Longitude (deg)'] % 360

souths_jdate = pd.read_csv("1980-2080-souths-jdate.csv", sep = ",", header=9)
souths_jdate.drop(souths_jdate.tail(4).index,inplace=True)
souths_jdate["UTC Julian date"] = souths_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
souths = pd.read_csv("1980-2080-souths-ecliptic-longitude.csv", sep = ",", header=15)
souths.drop(souths.tail(4).index,inplace=True)
souths = pd.merge(souths_jdate, souths)
souths["UTC calendar date"] = pd.to_datetime(souths["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
souths['Ecliptic Longitude'] = souths['Longitude (deg)'] % 360

equators_jdate = pd.read_csv("1980-2080-equators-jdate.csv", sep = ",", header=9)
equators_jdate.drop(equators_jdate.tail(4).index,inplace=True)
equators_jdate["UTC Julian date"] = equators_jdate["UTC Julian date"].str.replace(' JD UTC', '').astype(float)
equators = pd.read_csv("1980-2080-equators-ecliptic-longitude.csv", sep = ",", header=15)
equators.drop(equators.tail(4).index,inplace=True)
equators = pd.merge(equators_jdate, equators)
equators["UTC calendar date"] = pd.to_datetime(equators["UTC calendar date"], format='%Y-%m-%d %H:%M:%S.%f UTC')
equators['Ecliptic Longitude'] = equators['Longitude (deg)'] % 360

foo = []
for i in range(0,daily_lunar_eclip_longitude['Sidereal Cycles'].iloc[-1]):
    foo.append((daily_lunar_eclip_longitude[daily_lunar_eclip_longitude['Sidereal Cycles'] == i]['UTC Julian date'].iloc[0], daily_lunar_eclip_longitude[daily_lunar_eclip_longitude['Sidereal Cycles'] == (i+1)]['UTC Julian date'].iloc[0], i))
    siderealcyclehelper = pd.DataFrame(foo, columns=['start', 'end','cycle'])
    fulls.loc[((fulls['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (fulls['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    news.loc[((news['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (news['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    perigee.loc[((perigee['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (perigee['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    apogee.loc[((apogee['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (apogee['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    norths.loc[((norths['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (norths['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    souths.loc[((souths['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (souths['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    equators.loc[((equators['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (equators['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    starts_eclip_longitude.loc[((starts_eclip_longitude['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (starts_eclip_longitude['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i
    ends_eclip_longitude.loc[((ends_eclip_longitude['UTC Julian date'] >= siderealcyclehelper['start'].iloc[i]) & (ends_eclip_longitude['UTC Julian date'] < siderealcyclehelper['end'].iloc[i])), 'Sidereal Cycles'] = i

#####
#####
#####
#####

def tidecycles(number):
    first_cycle = number
    cycles = 1

    start = daily_lunar_eclip_longitude.loc[daily_lunar_eclip_longitude['Sidereal Cycles'] == (first_cycle)].index[0]
    end = daily_lunar_eclip_longitude.loc[daily_lunar_eclip_longitude['Sidereal Cycles'] == (first_cycle + cycles - 1)].index[-1]

    offset = daily_lunar_eclip_longitude['Sidereal Cycles'].iloc[start] - 1

    fullstart = fulls[fulls['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    fullend = fulls[fulls['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    newstart = news[news['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    newend = news[news['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    perigeestart = perigee[perigee['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    perigeeend = perigee[perigee['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    apogeestart = apogee[apogee['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    apogeeend = apogee[apogee['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    northsstart = norths[norths['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    northsend = norths[norths['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    southsstart = souths[souths['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    southsend = souths[souths['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    equatorsstart = equators[equators['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    equatorsend = equators[equators['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    startstart = starts_eclip_longitude[starts_eclip_longitude['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    startend = starts_eclip_longitude[starts_eclip_longitude['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]
    endstart = ends_eclip_longitude[starts_eclip_longitude['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    endend = ends_eclip_longitude[starts_eclip_longitude['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    hourlystart = hourly_ecl[hourly_ecl['UTC calendar date'] > daily_lunar_eclip_longitude['UTC calendar date'].loc[start]].index[0]
    hourlyend = hourly_ecl[hourly_ecl['UTC calendar date'] < daily_lunar_eclip_longitude['UTC calendar date'].loc[end]].index[-1]

    average_tide_hourly = sum(hourly_ecl["Prediction"])/len(hourly_ecl["Prediction"])
    hourly_pred_range = (hourly_ecl["Prediction"].max() - hourly_ecl["Prediction"].min())
    threshold_value = 0.3
    resized_threshold_value = ((threshold_value - average_tide_hourly)/hourly_pred_range)*abs(threshold_value - average_tide_hourly) + 0.5

    resized_prediction_hourly = ((hourly_ecl["Prediction"].loc[hourlystart:hourlyend] - average_tide_hourly)/hourly_pred_range)*abs(hourly_ecl["Prediction"].loc[hourlystart:hourlyend] - average_tide_hourly) + 0.5
    threshold = resized_prediction_hourly.loc[resized_prediction_hourly < resized_threshold_value]
    resized_prediction_hourly_starts = ((starts_eclip_longitude["Prediction"].loc[startstart:startend] - average_tide_hourly)/hourly_pred_range)*abs(starts_eclip_longitude["Prediction"].loc[startstart:startend] - average_tide_hourly) + 0.5
    resized_prediction_hourly_ends = ((ends_eclip_longitude["Prediction"].loc[endstart:endend] - average_tide_hourly)/hourly_pred_range)*abs(ends_eclip_longitude["Prediction"].loc[endstart:endend] - average_tide_hourly) + 0.5
        
    sun_t = daily_lunar_eclip_longitude['Solar Ecliptic Longitude'].loc[start:end]*np.pi/180
    full_t = fulls['Ecliptic Longitude'].loc[fullstart:fullend]*np.pi/180
    new_t = news['Ecliptic Longitude'].loc[newstart:newend]*np.pi/180
    perigee_t = perigee['Ecliptic Longitude'].loc[perigeestart:perigeeend]*np.pi/180
    apogee_t = apogee['Ecliptic Longitude'].loc[apogeestart:apogeeend]*np.pi/180
    norths_t = norths['Ecliptic Longitude'].loc[northsstart:northsend]*np.pi/180
    souths_t = souths['Ecliptic Longitude'].loc[southsstart:southsend]*np.pi/180
    equators_t = equators['Ecliptic Longitude'].loc[equatorsstart:equatorsend]*np.pi/180
    start_t = starts_eclip_longitude['Ecliptic Longitude'].loc[startstart:startend]*np.pi/180
    end_t = ends_eclip_longitude['Ecliptic Longitude'].loc[endstart:endend]*np.pi/180
    hourly_t = hourly_ecl['Ecliptic Longitude'].loc[hourlystart:hourlyend]*np.pi/180
    threshold_t = hourly_ecl['Ecliptic Longitude'].loc[hourlystart:hourlyend].loc[resized_prediction_hourly <= resized_threshold_value]*np.pi/180
    threshdot_t = hilo_ecl['Ecliptic Longitude'].loc[hourlystart:hourlyend].loc[resized_prediction_hourly <= resized_threshold_value]*np.pi/180

    # Circular Radii
    data_r = 6.5 - offset
    session_offset = 2
    sun_r = daily_lunar_eclip_longitude['Sidereal Cycles'].loc[start:end] + data_r
    full_r = fulls['Sidereal Cycles'].loc[fullstart:fullend] + data_r
    new_r = news['Sidereal Cycles'].loc[newstart:newend] + data_r
    perigee_r = perigee['Sidereal Cycles'].loc[perigeestart:perigeeend] + data_r
    apogee_r = apogee['Sidereal Cycles'].loc[apogeestart:apogeeend] + data_r
    norths_r = norths['Sidereal Cycles'].loc[northsstart:northsend] + data_r
    souths_r = souths['Sidereal Cycles'].loc[southsstart:southsend] + data_r
    equators_r = equators['Sidereal Cycles'].loc[equatorsstart:equatorsend] + data_r
    start_r = starts_eclip_longitude['Sidereal Cycles'].loc[startstart:startend] + data_r + resized_prediction_hourly_starts
    end_r = ends_eclip_longitude['Sidereal Cycles'].loc[endstart:endend] + data_r  + resized_prediction_hourly_ends
    hourly_r = hourly_ecl['Sidereal Cycles'].loc[hourlystart:hourlyend] + data_r + resized_prediction_hourly
    threshold_r = hourly_ecl["Sidereal Cycles"].loc[hourlystart:hourlyend].loc[resized_prediction_hourly <= resized_threshold_value] + data_r + resized_prediction_hourly.loc[resized_prediction_hourly <= resized_threshold_value]
    thresdot_r = hilo_ecl["Sidereal Cycles"].loc[hourlystart:hourlyend].loc[resized_prediction_hourly <= resized_threshold_value] + data_r + resized_prediction_hourly.loc[resized_prediction_hourly <= resized_threshold_value]

    

    # # Concentric Radii
    # # currently FUCKED UP
    # sun_r = daily_lunar_eclip_longitude['Radius'].loc[start:end] - offset + 7
    # full_r = fulls['Radius'].loc[fullstart:fullend] - offset + 7
    # new_r = news['Radius'].loc[newstart:newend] - offset + 7
    # perigee_r = perigee['Radius'].loc[perigeestart:perigeeend] - offset + 7
    # apogee_r = apogee['Radius'].loc[apogeestart:apogeeend] - offset + 7
    # norths_r = norths['Radius'].loc[northsstart:northsend] - offset + 7
    # souths_r = souths['Radius'].loc[southsstart:southsend] - offset + 7
    # equators_r = equators['Radius'].loc[equatorsstart:equatorsend] - offset + 7
    # start_r = starts_eclip_longitude['Radius'].loc[startstart:startend] - offset + 7 - 2
    # end_r = ends_eclip_longitude['Radius'].loc[endstart:endend] - offset + 7 - 2
    # hourly_r = hourly_ecl['Radius'].loc[hourlystart:hourlyend] - offset + resized_prediction2 + 7

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection = "polar")
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    # This fixes size...
    ax.set_ylim([0,10])

    # ax.plot(moon_t, moon_r, c="grey")
    ax.scatter(sun_t, sun_r, s=1000, facecolors="yellow",edgecolors="yellow", zorder=2)
    ax.scatter(full_t, full_r, s=1000, facecolors="white",edgecolors="black", zorder=3)
    ax.scatter(new_t, new_r, s=1000, facecolors="black",edgecolors="black", zorder=3)
    ax.scatter(apogee_t, apogee_r, s=500, facecolors="orange",edgecolors="black", zorder=3, marker="$\u0041$")
    ax.scatter(perigee_t, perigee_r, s=500, facecolors="orange",edgecolors="black", zorder=3, marker="$\u0050$")
    ax.scatter(norths_t, norths_r, s=500, facecolors="red",edgecolors="black", zorder=3, marker="$\u2B06$")
    ax.scatter(souths_t, souths_r, s=500, facecolors="red",edgecolors="black", zorder=3, marker="$\u2B07$")
    ax.scatter(equators_t, equators_r, s=500, facecolors="pink",edgecolors="black", zorder=3)
    # c = ax.scatter(theta, r, marker="$\u260A$")   # ascendings
    # c = ax.scatter(theta2, r2, marker="$\u260B$") # descendings
    # sessions
    # Need to prevent Feb 12 - Mar 10 2024 from happening. (360-1 session.)
    x = np.linspace(start_t, end_t)
    y = np.linspace(start_r, end_r)
    # ax.scatter(x, y, c="red");

    ax.plot(hourly_t, hourly_r, c="blue", zorder=0)
    ax.scatter(threshold_t, threshold_r, c="red", zorder=0)


    z_r = 4
    z_zoom=.3

    n=240
    t = np.linspace(0,360*np.pi/180, n)
    r = np.full(n, z_r+1)
    t1 = np.linspace(0,360*np.pi/180, n)
    r1 = np.full(n, z_r-1)
    ax.plot(t, r, 'black')
    ax.plot(t1, r1, 'black')
    for i in range(12):
        ax.plot(np.full(2,i*30*np.pi/180), np.linspace(z_r+1,z_r-1,2), "black")

    arr_img = plt.imread('Aries.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [15*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Taurus.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [45*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Gemini.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [75*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Cancer.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [105*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Leo.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [135*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Virgo.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [165*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Libra.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [195*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Scorpio.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [225*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Sagittarius.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [255*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Capricorn.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [285*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Aquarius.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [315*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    arr_img = plt.imread('Pisces.png', format='png')
    imagebox = OffsetImage(arr_img, zoom=z_zoom)
    xy = [345*np.pi/180, z_r]
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        boxcoords="offset points",
                        bboxprops=dict(alpha=0)
                        )
    ax.add_artist(ab)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title((daily_lunar_eclip_longitude.iloc[daily_lunar_eclip_longitude['Sidereal Cycles'].loc[start:end].index[0]]['UTC calendar date'].strftime('%B %d, %Y') + " to " + daily_lunar_eclip_longitude.iloc[daily_lunar_eclip_longitude['Sidereal Cycles'].loc[start:end].index[-1]]['UTC calendar date'].strftime('%B %d, %Y')), size=25)
    plt.savefig(str(number) + ".png")

tidecycles(576)

# end
