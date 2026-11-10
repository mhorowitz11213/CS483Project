SECTION ONE: PROJECT GOAL

Foreclosures remain a serious societal issues, causing homeloss and financial ruin. It has become a foremost objective for society to solve this issue, and to ensure that housing stability becomes ubiquitous across society. This is especially true of midwestern urban areas, which were hit particularly hard during the 2008 recession, and continue to lead in foreclosure numbers to this very actuality. [^1] Learning what causes foreclosures and in turn what can mitigate their numbers is the key objective of this research.

The goal of this project is to determine what causes foreclosures in midwest, with the interest of determining a. the specific factors that result in a foreclosure, and b. perhaps the features that can minimize such an event, helping people to stay in their homes. This research will utilize a variety of differing datasets, or differing structures to help in understanding the antecedents that lead to such an event.

SECTION TWO: DATA SOURCES

In order to capture the necessary features, the group turned to the following data sources (a description of which can be found in the next section)
    1. The Law Center for Better Housing Chicago Eviction Data: http://eviction.lcbh.org. To access the data simply click the download section.
    This data set is provided by the Law Center for Better Housing, a free legal aid society helping to stop foreclosures and evictions, and has been ranked as extremely reputable by Charity Navigator[^2]
    2. The Milwaukee Foreclosure GitHub repository: https://github.com/jdjohn215/milwaukee-foreclosures. To download just click on each file. 
    This repository is maintained by John D Johnson, a research fellow at Marquette University Law School's Lubar Center, who specializes in housing research[^3]. Sources for where he acquired specific data including the images, tabular data, and JSON files are documented in his README.
    3. MIT Places Dataset: Accessed by going to the MIT CSAIL GitHub Repository at https://github.com/CSAILVision/places365 and downloading the dataset from GitHub.
    This repository contains a multitude of images of places with names, we wills select images that correspond with slum or ruin as a foreclosure, and other labels for non foreclosure. Being hosted by CSAIL, one of the most reputable labs in computer science, we feel that this source is extremely reputable and worth using in our research.
    Direct download is available using the call: import torchvision.datasets as datasets and then writing dataset = datasets.Places365(root='./data', split='train-standard', small=True, download=True).
    4. Chicago Health Atlas: https://chicagohealthatlas.org/indicators/EVC.
    The Chicago Health Atlas website hosted by the City of Chicago, UIC Public Health, OSA Sprague Memorial Institute, Metopio, Schreiber Philanthropy, and Michael Reese has a variety of data on maps and tables that show evictions throughout the city, these will need to be scraped from the website. These is a well regarded cite supported by official city and civic institutions. 
    5. Illinois Courts: https://www.illinoiscourts.gov/approved-forms/circuit-court-standardized-forms-suites/eviction/.
    This dataset shows all court filings on evictions in the state of Illinois and will need to be scraped. The data comes from the official site for the Illinois Court System which handles all evictions.

SECTION THREE: DATASET DESCRIPTION 

The group has decided on a variety of datasets, that the group feels will capture the conditions that lead to foreclosures in midwestern urban areas. Essential to this project is utilizing large, heterogeneous datasets, that can capture a multitude of features leading to a foreclosed home. Each data structure type has its strengths and weakenesses. For this project, the group has decided on the following data set types:
    1. Tabular (Structured)
    2. Image (Unstructured)
    3. XML File (Semi-structured)
    4. HTML (Semi-structured)

Tabular:
The tabular data for this project comes from:
    The Law Center for Better Housing. This dataset consists of the following:
    -Eviction data: 4 CSVs, with each row per year for each corresponding year, tract, community area and ward, with the largest files containing over 8000 obs.
    -Census data: 7 CSVs, with census data at each geographic unity--city, community area, ward, tract--for Chicago.
    The Milwaukee Foreclosure GitHub repository. This repository contains five CSV files on Milwaukee foreclosures at each census level plus a geocoded foreclosure list with over 36,000 observations
Image:
    The image data comes from part of the MIT Places repository.



[^1] D'Amico, A (2026), "'Zombie' Foreclosures Are Most Common in These Midwestern Cities", Realtor.com, Available: https://www.msn.com/en-us/urban-infrastructure/housing-and-urban-development/zombie-foreclosures-are-most-common-in-these-midwestern-cities/ar-AA2bhFft?ocid=BingNewsVerp Accessed November 2 2026.
[^2] Charity Navigator (nd.), "Law Center for Better Housing", Available:https://www.charitynavigator.org/ein/363134577 Accessed November 2 2026.
[^3] Marquette University (n.d.), "Lubar Center Team", Available: https://law.marquette.edu/lubar-center/lubar-center-team Accessed November 2 2026

Image:
The images for this project come from the New York Times. The dataset consists of the following:
    -43,634 Properties on the brink of foreclosure in 2014. Can be CSV or folder of likely JPEGs.
    -Data:


[^1] Kissane, E (2014), "The NYT's Detroit Foreclosure Interactive", OpenNews Source, Available: opennews.org Accessed November 4 2026.

XML File:
The XML file (SVG vector image) for this project comes from the Chicago Health Atlas. The dataset consists of the following:
    -Eviction count (EVC), eviction rate (EVR), geographic markers, and time indicators.
    -Data:


[^1] Chicago Health Atlas (2026), "Evictions," City of Chicago Interactive Public Data Dashboard, Available: https://chicagohealthatlas.org/indicators/EVC?tab=chart Accessed November 4 2026.

HTML:
The HTML file for this project comes from the Eviction Research Network in collaboration with U.C. Berkeley's Urban Displacement Project. The dataset consists of the following:
    -Eviction filing counts, demographic/racial filing rate disparities, and geographic markers.
    -Data:


[^1] Moore, H., Cohen, A., and Thomas, T. (2023), "Cook County Illinois Eviction Filings," Eviction Research Network & U.C. Berkeley Urban Displacement Project, Available: https://evictionresearch.net/illinois/ Accessed November 4 2026.

Challenges: Texting matching typos such as "St." vs "Street" and missing apartment numbers can present challenges. Also, if boundaries have shifted for communities, that can cause problems as well.

Cleaning, Transformation, Integration: We plan to clean through removing duplicates and identifying weird discrepancies in the data. We will transform anything necessary and integrate through merging the data properly.

SECTION 4: Veracity of Data
Our data comes from reliable academic, government, and legal institutions. However, the housing and court sources rely on municipal clerk entries and legal templates built for administration rather than statistical analysis. 

Missing Values, Duplicates, & Inconsistencies
    LCBH, Milwaukee, Health Atlas: Contain duplicate filings, unstandardized addresses and landlord LLC names, and not much data in low-population areas

    Places365: Contains near duplicate images, corrupted files, and very similar  labels such as apartment_building vs. residential_neighborhood for the same house

    Illinois Court Forms: dates contain unstructured text and checkboxes

Errors, Outdated Info, & Biases

    Bias: Court records only capture formal filings and don't include informal evictions like lockouts or tenants leaving early. Data from 2020–2022 is also  skewed by COVID-19 and the Milwaukee dataset ends in 2022.

    Measurement & Errors: Health Atlas rates rely on Census ACS estimates that have margins of error, Places365 has web-scraping geographic biases, and handwritten court forms can have errors.

Cleaning & Preprocessing Plan

    Normalize addresses and combine multi defendant cases by (Case_ID, Address, Date)

    Filter out commercial properties and add time indicators to separate COVID-19  years from baseline trends

    Clean Images & Forms: Drop corrupted/duplicate images using perceptual hashing and resize tensors to 224×224. Use PDF parsers with strict validation to normalize dates to YYYY-MM-DD

SECTION FIVE: VALUE OF DATA

Our data is valuable because it shows where evictions happen, which demographics they affect, and what might help prevent them. Evictions in general can also cause immense financial strain on households that can become homeless or jobless due to the loss of an address for themselves and their families.

This data is also valuable because different legal groups throughout Chicago can also use these datasets to help them decide which neighborhoods to focus on first, such as where to offer free legal help, rental assistance, or tenant education. The LCBH data also breaks down data by ward, which allows for local members of the City Council of Chicago to visually see eviction rates in the areas they represent and push towards the local assembly for valid solutions that can help the people within areas that statistically need more support.

The data provides strong insight into clusters of neighborhoods with high eviction rates, such as the South Shore, which had the most filings and the highest rate in 2019. The addition of the HUD voucher data also lets us test whether areas with many housing-voucher households have different eviction rates compared to areas with fewer housing-voucher households. We can also see trends over years of data to see if eviction filings changed the fastest throughout a period of time. 

The data is also valuable in the aspects of different formats of data visualization. For example, the MIT Places365 dataset contains about 1.8 million labeled images across 365 types of scenes. Adding different styles of datasets such as court filings, census data, voucher counts, and images gives a broader representation, as it allows us to see all of the sources against each other in a clearer format. This also makes our data more accurate, as we are not biasing our data into one specific dataset that may trend towards one specific opinion. 
