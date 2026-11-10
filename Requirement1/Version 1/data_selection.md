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
XML File:
    The XML files consists of SVG bector image, and consists of:
    ~Eviction count (EVC), eviction rate (EVR), geographic, and time indicators. In addition, further information can be scraped and parsed into JSON files comes from the Chicago Health Atlas with code from Ryan Zomorrodi https://github.com/ryanzomorrodi/healthatlas a research assistant at UIC--one of the groups running the Chicago Health Atlas. Size of data is variable and will be scraped this week.
HTML:
    The HTML data is data of court filings found at the Illinois Court Website. The data contains all evictions, which the group will process as foreclosures. The size is variable.


SECTION FOUR: ANALYSIS OF THE FIVE V's

1. Veracity:
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
    
2. Velocity:
Our research uses datasets from several sources. Velocity is the speed at which data is collected, updated, and made available. These sources are not updated at the same rate which can affect how quickly we identify and analyze changes in trends.
    The datasets used in this research are all fairly large:
    1. LCBH datasets contain multiple files over over 8,000 records and 36 columns.
    2. For LCBH and Milwaukee Eviction dataset, they are not updated in real time and are added as eviction cases are filed and         processed. This means that there may be delays when an eviction occurs and when the information becomes available.
    3. Health Atlas: This data is updated periodically. American Community Survey estimates are released annually which means that      recent changes may not be seen immediately. 
    4. MIT: This MIT dataset has a static image dataset containing approximately millions of labeled images across categories. It       does not continuously receive new information, so its images may not represent recent neighborhood changes.
    4. Illinois Courts: This website periodically updates its eviction forms. We will track approval dates and make sure we use         the appropriate versions when extracting information.

Our data is updated at different frequencies. This means some information may be more recent than others. This can make it difficult to compare datasets for the same year or identify immediate changes. To address these limitations, we shall consider the publication dates and time periods of the datasets when analyzing them. We will align the datasets by year whenever possible. We will identify gaps between lastest data. Although our datasets are not updated in realtime, we have enough historical data.

3. Variety
For variety, there are different types, formats, and structures of data. Our project combines education, government, and legal sources to identify eviction patterns, neighborhood demographics, housing conditions, and more.
    1. LCBH and Milwaukee: These datasets primarily contain structured tabular data. These records are generally organized in CSV         or spreadsheet formulas making them suitable for statistical analysis and data visualization.
    2. Health Atlas: These are structured demographic and socioeconomic information like population estimates and household               characteristics. This allows us to compare eviction activity with characteristics of different communities.
    3. MIT: There is unstructured visual data. It has visual information that can't be shown by numbers alone.
    4. Illinois Courts: These sources contain semi-structured legal documents in PDF formatting.

Our data variety allows us to combine structured, semi-structured, and unstructured information to develop a better understanding of eviction patterns. We can also explore data through textual characteristics and visual through the images.

4. Veracity
For veracity, we are looking at quality, accuracy, consistency, and reliability of our data. These sources provide valuable information, but we need to take advantage of it through our case for statistical analysis. We must identify information that is most important to us.
    1. LCBH, Milwaukee, and Health Atlas: These datasets may contain duplicate eviction filings and variations in landlord names          due to different LLC or property management companies. Some geographic areas may also have limited information like                neighborhoods with small populations.
    2. MIT: This dataset may contain duplicate images, corrupted files, and inconsistent scene labels. Image classification may be        less consistent depending on how images are labelled and what is being analyzed.
    3. Illinois Courts: Extracting information from PDF files may introduce errors that we need to consider.

To improve reliability, we will need to make sure our data is clean. We will need to normalize data and identify duplicates. Cleaning, Standardizing, and validating the data to reduce errors and ensure data reliability is important.

5. Value
    Our data is valuable because it shows where evictions happen, which demographics they affect, and what might help prevent them. Evictions in general can also cause immense financial strain on households that can become homeless or jobless due to the loss of an address for themselves and their families.

    This data is also valuable because different legal groups throughout Chicago can also use these datasets to help them decide which neighborhoods to focus on first, such as where to offer free legal help, rental assistance, or tenant education. The LCBH data also breaks down data by ward, which allows for local members of the City Council of Chicago to visually see eviction rates in the areas they represent and push towards the local assembly for valid solutions that can help the people within areas that statistically need more support.

    The data provides strong insight into clusters of neighborhoods with high eviction rates, such as the South Shore, which had the most filings and the highest rate in 2019. The addition of the HUD voucher data also lets us test whether areas with many housing-voucher households have different eviction rates compared to areas with fewer housing-voucher households. We can also see trends over years of data to see if eviction filings changed the fastest throughout a period of time. 

    The data is also valuable in the aspects of different formats of data visualization. For example, the MIT Places365 dataset contains about 1.8 million labeled images across 365 types of scenes. Adding different styles of datasets such as court filings, census data, voucher counts, and images gives a broader representation, as it allows us to see all of the sources against each other in a clearer format. This also makes our data more accurate, as we are not biasing our data into one specific dataset that may trend towards one specific opinion. 

SECTION FIVE: DATA PREPARATION
Cleaning, Transformation, Integration: We plan to clean through removing duplicates and identifying weird discrepancies in the data. We will transform anything necessary and integrate through merging the data properly.

SECTION SIX: LIMITATIONS AND RISKS
Challenges: Texting matching typos such as "St." vs "Street" and missing apartment numbers can present challenges. In multi-unit buildings the missing apartment numbers are more tricky. This is because many different family evictions get combined into a single point on or even dropped altogether. Another risk is that  boundaries don’t stay the same over time. When comparing data across a long periods of time there will be shifting Census reports and new neighborhood boundaries which will leave us to estimate where one area ends and another begins.

Another major challenge is that court records don't tell the whole story. Court numbers include formal filings but they don't account for  informal things such as  landlords changing locks, shutting off power, or tenants leaving early to avoid having a court filing on their record. Even when a case reaches court a filing doesn't guarantee someone actually moves out. Still, that household will always have an eviction listed in background checks. There are also abandoned "zombie" foreclosures that slip through when the registry audits. There are month long delays in city reporting and raw data doens't account for that, which might make it seem like one month was really bad. On the other side, a sudden improvement in neighborhood stability could just be a hold up at courthouse.




[^1] D'Amico, A (2026), "'Zombie' Foreclosures Are Most Common in These Midwestern Cities", Realtor.com, Available: https://www.msn.com/en-us/urban-infrastructure/housing-and-urban-development/zombie-foreclosures-are-most-common-in-these-midwestern-cities/ar-AA2bhFft?ocid=BingNewsVerp Accessed November 2 2026.
[^2] Charity Navigator (nd.), "Law Center for Better Housing", Available:https://www.charitynavigator.org/ein/363134577 Accessed November 2 2026.
[^3] Marquette University (n.d.), "Lubar Center Team", Available: https://law.marquette.edu/lubar-center/lubar-center-team Accessed November 2 2026
[^4] Kissane, E (2014), "The NYT's Detroit Foreclosure Interactive", OpenNews Source, Available: opennews.org Accessed November 4 2026.
[^5] Chicago Health Atlas (2026), "Evictions," City of Chicago Interactive Public Data Dashboard, Available: https://chicagohealthatlas.org/indicators/EVC?tab=chart Accessed November 4 2026.
[^6] Moore, H., Cohen, A., and Thomas, T. (2023), "Cook County Illinois Eviction Filings," Eviction Research Network & U.C. Berkeley Urban Displacement Project, Available: https://evictionresearch.net/illinois/ Accessed November 4 2026.






