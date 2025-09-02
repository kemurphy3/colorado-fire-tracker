# Colorado Fire Watch

A wildfire risk prediction tool that combines NEON's detailed ecological data with satellite imagery to map fire danger across Colorado.

## What this is

Living in Colorado, wildfire season feels inevitable. Most fire risk models look at weather patterns and basic vegetation indices, but they miss a lot of the ground-level detail that actually matters.

This project tries something different. NEON's Aerial Observation Platform (AOP) team flies over the field sites in a pushbroom pattern using its sensors to capture hyperspectral, LiDAR, and high-resolution camera imagery data at an incredible resolution. The trade off is that the flightboxes for each site are typically only 10x10kms. In order to properly scale the fine NEON data, I'm developing a crosswalk to translate it to satellites like Sentinel-2 and MODIS that cover the whole state.

## Why I'm building this

A few reasons:
- The Cameron Peak Fire in 2020 burned right through areas near NEON's NIWO site. We have very useful before/after data
- Traditional NDVI (vegetation greenness) completely misses drought-stressed conifers that still look green but are basically tinderboxes
- I work with NEON data professionally and kept thinking "we could do something useful with this"

## Current status

Still early days. Right now I am:
- Building the crosswalk between NEON's hyperspectral bands and Sentinel-2's multispectral ones
- Setting up a basic web interface to visualize risk levels

## What's coming

**Soon:**
- Basic risk map for Colorado using the NEON-to-Sentinel crosswalk
- Validation against the Cameron Peak and East Troublesome fires

**Eventually:**
- MODIS integration for daily updates (Sentinel only passes over every 5 days)
- Expand to other western states with NEON sites
- Add weather data, fuel moisture, etc

**Maybe someday:**
- Real-time smoke plume detection
- Prescribed burn planning tools
- API for other developers

## Tech stack

- Python for all the heavy lifting (rasterio, xarray, scikit-learn)
- PostGIS for spatial data
- FastAPI for the backend
- Still deciding on frontend (probably just React + Mapbox)

## Data sources

- NEON AOP: Hyperspectral + LiDAR from their Colorado sites (NIWO, RMNP, CPER)
- Sentinel-2: 10m resolution, every 5 days
- MODIS: Daily coverage but coarser (250m-1km)
- Historical fire perimeters from MTBS and GeoMAC

## Contributing

Open an issue or submit a PR if you're interested in contributing.

## License

MIT - use it however you want.
