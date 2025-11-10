from ..dataloaders.pubhealth_loader import Loader as PubHealthLoader
from ..dataloaders.osf_loader import OSFLoader


def load_data():
    all = []

    # pubhealth = PubHealthLoader()
    # all.extend(pubhealth.extract())
    osf = OSFLoader()
    all.extend(osf.extract())
    print(all[:5])


load_data()
