This contains data from an experimental HOPS run of the EHTC 2024 data.
This data has not been thoroughly vetted for issues so there may be
severe problems.

The data is mixed polarization meaning ALMA has not been polconverted even though the labels in the uvfits file say it is circular.

The ALMA polarization mapping is
L -> X/H (horizontal)
R -> Y/V (vertical)

Data Qualities:
 - band 2 data (230 GHz track)
 - The data has not been ALMA rotated. Meaning the ALMA feed has a 45 degree offset.
 - The apriori cal is based off of weather data from Merra 2 and telescope metadata that lives in the ngehtsim package. Gains are likely only good to 50 in general. This means that total flux measurements can be significantly biased.
 - Due to the data being mixed polarization we have not applied feed rotation calibration. Do not try to fit Stokes I only unless you correct for this and flag ALMA
 - The data has not been averaged beyond the channels and the 0.4s correlator dump. This is quite useful for inspecting residual issues in the data, e.g., bandpass effects or coherence loss.
