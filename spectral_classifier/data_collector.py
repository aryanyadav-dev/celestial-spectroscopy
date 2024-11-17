import numpy as np
from astroquery.sdss import SDSS
from tqdm import tqdm

class SpectralDataCollector:
    """Class to collect spectral data from various astronomical databases"""
    
    def __init__(self, save_dir='./spectral_data'):
        self.save_dir = save_dir
    
    def fetch_sdss_spectra(self, num_samples=1000):
        """
        Fetch spectral data from SDSS using updated query method
        """
        print("Fetching SDSS spectra...")
        spectra = []
        labels = []
        
        try:
            query = """
            SELECT TOP {0}
                s.plate, s.fiberID, s.mjd, s.class
            FROM SpecObj s
            WHERE s.class IN ('STAR', 'GALAXY')
            AND s.zWarning = 0
            """.format(num_samples)
            
            results = SDSS.query_sql(query)
            
            if results is None or len(results) == 0:
                print("No results from SDSS query. Using synthetic data instead.")
                return self._generate_synthetic_data(num_samples)
            
            print(f"Found {len(results)} spectra. Downloading...")
            
            for entry in tqdm(results[:num_samples]):
                try:
                    spec = SDSS.get_spectra(plate=entry['plate'], 
                                          fiberID=entry['fiberID'],
                                          mjd=entry['mjd'])
                    
                    if spec is not None and len(spec) > 0:
                        flux = spec[0][1].data['flux']
                        if len(flux) != 1024:
                            flux = np.interp(np.linspace(0, len(flux), 1024), np.arange(len(flux)), flux)
                        spectra.append(flux)
                        labels.append(entry['class'].lower())
                except Exception as e:
                    print(f"Error fetching spectrum: {e}")
                    continue

            if len(spectra) == 0:
                print("No valid spectra downloaded. Using synthetic data instead.")
                return self._generate_synthetic_data(num_samples)

            return np.array(spectra), np.array(labels)
        except Exception as e:
            print(f"Error accessing SDSS: {e}")
            return self._generate_synthetic_data(num_samples)

    def _generate_synthetic_data(self, num_samples):
        """Generate synthetic spectral data when real data is unavailable"""
        print("Generating synthetic spectral data...")
        spectra = []
        labels = []
        patterns = {
            'star': [(4861, 100), (6563, 150)],  # H-alpha and H-beta lines
            'galaxy': [(3727, 80), (5007, 120)],  # OII and OIII lines
        }
        
        for _ in tqdm(range(num_samples)):
            obj_type = np.random.choice(['star', 'galaxy'])
            spectrum = np.zeros(1024)
            for wavelength, intensity in patterns[obj_type]:
                idx = wavelength * 1024 // 10000
                spectrum[idx-5:idx+5] += np.random.normal(intensity, 10)
            spectrum += np.random.normal(0, 5, 1024)
            redshift_factor = np.random.uniform(0.0, 0.05)
            spectrum = np.interp(np.arange(1024), np.arange(1024) * (1 + redshift_factor), spectrum)

            spectra.append(spectrum)
            labels.append(obj_type)

        return np.array(spectra), np.array(labels)
