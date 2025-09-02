# Colorado Fire Watch

A wildfire risk prediction tool that combines NEON's detailed ecological data with satellite imagery to map fire danger across Colorado.

## What this is

Living in Colorado, you can't ignore wildfire season. The smell of smoke, the orange skies, the evacuation warnings - it's become part of life here. Most fire risk models look at weather patterns and basic vegetation indices, but they miss a lot of the ground-level detail that actually matters.

This project tries something different. NEON flies planes over their field sites collecting incredibly detailed data - hyperspectral imagery that sees way beyond RGB, LiDAR that maps vegetation structure down to individual trees. The problem? They only cover small areas. So I'm working on translating those insights to satellites like Sentinel-2 and MODIS that cover the whole state.

Think of it like this: NEON shows us what healthy vs stressed vegetation *really* looks like at a molecular level. We then teach satellites to recognize those same patterns in their lower-resolution data. It's not perfect, but it's better than pretending all "green" pixels are the same.

## Why I'm building this

A few reasons:
- The Cameron Peak Fire in 2020 burned right through areas near NEON's NIWO site. We have before/after data at insane resolution.
- Traditional NDVI (vegetation greenness) completely misses drought-stressed conifers that still look green but are basically tinderboxes
- I work with NEON data professionally and kept thinking "we could do something useful with this"

## Current status

Still early days. Right now I'm:
- Getting the NEON AOP data pipeline working (these files are massive)
- Building the crosswalk between NEON's hyperspectral bands and Sentinel-2's multispectral ones
- Setting up a basic web interface to visualize risk levels

## What's coming

**Soon:**
- Basic risk map for Colorado using the NEON-to-Sentinel crosswalk
- Validation against the Cameron Peak and East Troublesome fires

**Eventually:**
- MODIS integration for daily updates (Sentinel only passes over every 5 days)
- Expand to other western states with NEON sites
- Add weather data, fuel moisture, maybe even power line locations

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
