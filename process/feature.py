# https://github.com/pregHosh/torchchem/blob/12e5f03e9f1a2cf2b24c3fdc6478720c3dca112b/torchdrug/data/feature.py

import warnings

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

#from torchdrug.core import Registry as R

# orderd by perodic table
atom_vocab = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Cu",
    "Zn",
    "Ge",
    "As",
    "Se",
    "Br",
    "Sn",
    "I",
]

atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(8)
num_hs_vocab = range(7)
formal_charge_vocab = range(-6, 9)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# orderd by molecular mass
residue_vocab = [
    "GLY",
    "ALA",
    "SER",
    "PRO",
    "VAL",
    "THR",
    "CYS",
    "ILE",
    "LEU",
    "ASN",
    "ASP",
    "GLN",
    "LYS",
    "GLU",
    "MET",
    "HIS",
    "PHE",
    "ARG",
    "TYR",
    "TRP",
]


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(
                "Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab)
            )
        feature[index] = 1

    return feature


# TODO: this one is too slow
#@R.register("features.atom.default")
def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetChiralTag(), chiral_tag_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetNumRadicalElectrons(), num_radical_vocab)
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


#@R.register("features.atom.default_extra")
def atom_default_extra(atom):
    """Default atom feature.

    Features:

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    num_hs_vocab = range(5)
    valence_vocab = range(8)
    return (
        # onehot(atom.GetChiralTag(), chiral_tag_vocab)
        onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), valence_vocab)
        # + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + onehot(atom.GetTotalNumHs(True), num_hs_vocab)
        # + onehot(atom.GetNumRadicalElectrons(), num_radical_vocab)
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


#@R.register("features.atom.default_condense")
def atom_default_condense(atom):
    """Default atom feature.

    Features:

        GetTotalDegree(): the degree of the atom in the molecule including Hs

        GetTotalDegree(): the degree of the atom in the molecule including Hs

        GetTotalValence(): the total valence (explicit + implicit) of the atom

        GetTotalNumHs(): the total number of Hs (explicit and implicit) on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        # [atom.GetChiralTag()] # not specified
        [atom.GetTotalDegree()]
        + [atom.GetTotalValence()]
        + [atom.GetFormalCharge()]
        + [atom.GetTotalNumHs(True)]
        # + [atom.GetTotalNumHs()] # somehow return all 0
        # + [atom.GetNumRadicalElectrons()]
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


#@R.register("features.atom.center_identification")
def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


#@R.register("features.atom.synthon_completion")
def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        IsInRing(): whether the atom is in a ring

        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size

        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + [
            atom.IsInRing(),
            atom.IsInRingSize(3),
            atom.IsInRingSize(4),
            atom.IsInRingSize(5),
            atom.IsInRingSize(6),
            atom.IsInRing()
            and (not atom.IsInRingSize(3))
            and (not atom.IsInRingSize(4))
            and (not atom.IsInRingSize(5))
            and (not atom.IsInRingSize(6)),
        ]
    )


#@R.register("features.atom.symbol")
def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)


#@R.register("features.atom.explicit_property_prediction")
def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + [atom.GetIsAromatic()]
    )


#@R.register("features.atom.property_prediction")
def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True)
        + [atom.GetIsAromatic()]
    )


#@R.register("features.atom.position")
def atom_position(atom):
    """
    Atom position in the molecular conformation.
    Return 3D position if available, otherwise 2D position is returned.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]


#@R.register("features.atom.pretrain")
def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + onehot(
        atom.GetChiralTag(), chiral_tag_vocab
    )


#@R.register("features.atom.residue_symbol")
def atom_residue_symbol(atom):
    """Residue symbol as atom feature. Only support atoms in a protein.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        GetResidueName(): one-hot embedding for the residue symbol
    """
    residue = atom.GetPDBResidueInfo()
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + onehot(
        residue.GetResidueName() if residue else -1, residue_vocab, allow_unknown=True
    )


#@R.register("features.bond.default")
def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond

        GetStereo(): one-hot embedding for the stereo configuration of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return (
        onehot(bond.GetBondType(), bond_type_vocab)
        + onehot(bond.GetBondDir(), bond_dir_vocab)
        + onehot(bond.GetStereo(), bond_stereo_vocab)
        + [int(bond.GetIsConjugated())]
    )


#@R.register("features.bond.length")
def bond_length(bond):
    """
    Bond length in the molecular conformation.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]


#@R.register("features.bond.property_prediction")
def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated

        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + [
        int(bond.GetIsConjugated()),
        bond.IsInRing(),
    ]


#@R.register("features.bond.pretrain")
def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + onehot(
        bond.GetBondDir(), bond_dir_vocab
    )


#@R.register("features.residue.symbol")
def residue_symbol(residue):
    """Symbol residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return onehot(residue.GetResidueName(), residue_vocab, allow_unknown=True)


#@R.register("features.residue.default")
def residue_default(residue):
    """Default residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return residue_symbol(residue)


#@R.register("features.molecule.ecfp")
def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)


##@R.register("features.molecule.default")
def molecule_default(mol):
    """Default molecule feature."""
    return ExtendedConnectivityFingerprint(mol)


ECFP = ExtendedConnectivityFingerprint

try:
    import networkx as nx
    import numpy as np
    import scipy.spatial
    from ase.data import covalent_radii
    from ase.data.vdw_alvarez import vdw_radii
    from cell2mol.elementdata import ElementData
    from morfeus import SASA

    # from libarvo import atomic_vs

    def get_radii(z, radii="covalent"):
        if radii == "covalent":
            return np.array([covalent_radii[i] for i in z], dtype=float)
        if radii == "vdw":
            return np.array([vdw_radii[i] for i in z], dtype=float)

    def get_other_features(z):
        ed = ElementData()
        sym = [ed.elementsym[zi] for zi in z]
        ve = [ed.valenceelectrons[symi] for symi in sym]
        en = [ed.ElectroNegativityPauling[symi] for symi in sym]
        return ve, en

    def get_dm(z, coordinates):
        n = len(z)
        dm = scipy.spatial.distance.pdist(coordinates.numpy())
        return dm

    def get_am(z, coordinates, radii, dm, scale_factor=1.15):
        n = len(z)
        am = np.zeros((n, n), dtype=float)
        row, col = np.triu_indices(n, 1)
        rm = scale_factor * scipy.spatial.distance.pdist(
            radii.reshape(-1, 1), metric=lambda x, y: x + y
        )
        am[row, col] = am[col, row] = dm - rm
        return am < 0

    def bin(arr, n_bins=5):
        intervals = np.linspace(min(arr), max(arr), n_bins)
        return np.digitize(arr, intervals)

    def atom_geom(z, coords, bin=True):
        """
        Compute the geometric features of the atoms in the molecule.
        params:
        z: np.array of atomic numbers
        coords: torch.tensor of atomic coordinates

        returns:
        geom_node_feat: torch.tensor of geometric features of the atoms in the molecule
        - degree: Total degree of the atom
        - a_volume: atomic volume
        - a_surface: atomic surface
        - occ_v: occupied volume
        - ve: valence electrons
        - en: pauling electronegativity

        """
        TOL = 0.2
        cov_radii = get_radii(z, radii="covalent")
        vdw_radii = get_radii(z, radii="vdw")
        cov_dm = get_dm(z, coords)
        cov_am = get_am(z, coords, cov_radii, cov_dm, scale_factor=1.15)
        G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
        degree = [val for (i, val) in G.degree()]
        if not (cov_dm > TOL).all():
            print("Some atoms are incredibly close to each other!")

        # a_volume, a_surface = atomic_vs(coords, vdw_radii)
        sasa = SASA(z, coords.numpy(), vdw_radii, probe_radius=0.0, density=0.01)
        sa_volume = np.fromiter(sasa.atom_volumes.values(), dtype=float)
        sa_surface = np.fromiter(sasa.atom_areas.values(), dtype=float)
        ve, en = get_other_features(z)
        ref_v = np.pi * (4 / 3) * vdw_radii**3
        occ_v = (
            ref_v - sa_volume
        ) / ref_v  # This one looks promising since its normalized by definition.

        ## It might happen than rounding to int makes entire feature 0 or 1 or something like that which sucks
        ## an alternative is definining bins and then binning the array
        # bins = np.array([-1.0,-0.1, 0.0, 0.1, 1.0])
        # occ_v = np.digitize(occ_v, bins) # Now this will contain indices of the bin to which the elements belong
        ## We could also figure out how to split automatically
        # occ_v = bin(occ_v)

        # if bin:  # We make everything integers
        #     a_volume = a_volume.astype(int)
        #     a_surface = a_surface.astype(int)
        #     occ_v = occ_v.astype(int)
        #     en = en.astype(int)

        geom_node_feat = torch.tensor(
            [
                degree,
                sa_volume / 10,
                sa_surface / 10,
                occ_v / 10,
                ve,
                en,
            ]
        ).T
        return geom_node_feat

except ImportError as m:
    print(
        f"Some dependencies are not installed, cannot featurize with 3D features. Error message:\n{m}"
    )
    atom_geom = None

__all__ = [
    "atom_default",
    "atom_center_identification",
    "atom_synthon_completion",
    "atom_symbol",
    "atom_explicit_property_prediction",
    "atom_property_prediction",
    "atom_position",
    "atom_pretrain",
    "atom_residue_symbol",
    "atom_geom",
    "bond_default",
    "bond_length",
    "bond_property_prediction",
    "bond_pretrain",
    "residue_symbol",
    "residue_default",
    "ExtendedConnectivityFingerprint",
    "molecule_default",
    "ECFP",
]
