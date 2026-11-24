# Citation ledger

Verified sources behind the claims in this project's cards and plans (AGENTS.md §7). For each source the ledger
records:

- what was read;
- the exact claim the source supports, with every number kept with its denominator;
- when that differs from the original paper's scope, the patient group, timing and field strength.

A claim that is not listed here, or is listed as not verified, must not be stated as fact.

Access marks:
- **FT**: full text read.
- **Abs**: abstract only, through PubMed.
- **Sec**: seen only through another source.
- **NV**: not verified.

Unless an entry says otherwise, Claude Code checked it with the PubMed tools on 2026-11-24. No clinician has
reviewed these readings.

## 1. Early postoperative MRI after brain resection

Context: the deep-research report
`docs/Early Postoperative MRI After Adult Glioma Resection_ Cavity Appearance, Tumour Mimics and Implications for Synthetic Data.md`
(Claude chat, 2026-11-24) and the checks below. §1.6 lists the corrections to that report.

### 1.1 What fills the cavity

- Johnstad C, Reinertsen I, Bouget D, Sagberg LM, Strand PS, Solheim O. Incidence, risk factors, and clinical
  implications of postoperative blood in or near the resection cavity after glioma surgery. Brain Spine
  2024;4:102818. doi:10.1016/j.bas.2024.102818. **FT** (PMC11081780).
  - 292 diffuse glioma resections (WHO grade 2–4), single centre, MRI 12–48 h after surgery.
  - Intradural blood of 0.1 mL or more in 184/292 (63%), median 0.4 mL. Extradural blood in 212/292 (73%), median
    3.0 mL.
  - Blood is defined as T1 iso- to hyperintense with T2 hypointensity.
  - Intradural blood volume grows with tumour volume (quartile medians 0.1 to 0.9 mL) and with blood loss. It is
    not associated with location or WHO grade.
  - Quote: "liquid blood, bloody CSF, and bloody saline exhibit different signals than coagulated blood."
  - Limits: a single rater with no reliability check; haemostatic material may be counted as blood.
- Meyding-Lamadé U, Forsting M, Albert F, Kunze S, Sartor K. Accelerated methaemoglobin formation: potential pitfall
  in early postoperative MRI. Neuroradiology 1993;35:178–180. doi:10.1007/BF00588487. **Abs**.
  - 24/54 (44%) showed areas of increased signal on unenhanced T1 soon after surgery.
  - Hydrogen peroxide, used for haemostasis, accelerated methaemoglobin formation in vitro.
- Forsyth PA et al. Prospective study of postoperative magnetic resonance imaging in patients with malignant
  gliomas. J Clin Oncol 1997;15:2076–2081. doi:10.1200/JCO.1997.15.5.2076. **Abs**.
  - 17 patients scanned on days 1, 3, 5, 7, 14 and 21.
  - Methaemoglobin at some time point in all 17; it interfered with residual-tumour assessment in 6 (35%).
  - Non-tumoural marginal enhancement in 8 (53%), maximal on days 5–14. The authors recommend days 3–5.
- Bradley WG. MR appearance of hemorrhage in the brain. Radiology 1993;189:15–26.
  doi:10.1148/radiology.189.1.8372185. **Abs**.
  - Five stages: hyperacute (long T1, long T2), acute (short T2), early subacute (short T1, short T2), late subacute
    (short T1, long T2), chronic (short T2).
  - Susceptibility effects are stronger on gradient-echo images.
  - Describes spontaneous haematoma, not surgical blood.
- Melhem ER, Jara H, Eustace S. Fluid-attenuated inversion recovery MR imaging: identification of protein
  concentration thresholds for CSF hyperintensity. AJR 1997;169:859–862. doi:10.2214/ajr.169.3.9275912. **Abs**.
  - Phantoms and volunteers, not a postoperative study.
  - Saline containing protein becomes FLAIR-hyperintense to brain above a threshold that depends on echo time:
    250 mg/dl at TE 110 ms down to 95 mg/dl at TE 250 ms.
- Sato N et al. Postoperative changes in the brain: MR imaging findings in patients without neoplasms. Radiology
  1997;204:839–846. doi:10.1148/radiology.204.3.9280269. **Abs**.
  - Patients: temporal lobectomy for epilepsy, not tumour.
  - Air on 31/35 (89%) of images in days 0–4, none after day 5.
  - Extra-axial fluid on all 46 images in the first 30 days, gone by day 51.
  - Margin enhancement:
    - days 0–5: thin linear in 16/18 (89%);
    - days 6–29: thick linear or nodular in 11/18 (61%);
    - none after day 30.
  - Dural enhancement on 96/97 images.
- Henegar MM, Moran CJ, Silbergeld DL. Early postoperative magnetic resonance imaging following nonneoplastic
  cortical resection. J Neurosurg 1996;84:174–179. doi:10.3171/jns.1996.84.2.0174. **Abs**.
  - Patients: epilepsy, 11 patients scanned at 17–28 h.
  - Extra-axial fluid, air or blood in all 11.
  - Resection-bed enhancement in 7/11 within 17 h, usually linear.
  - Dural enhancement next to the site in all 11.
- Kothbauer KF et al. Foreign body reaction to hemostatic materials mimicking recurrent brain tumor. J Neurosurg
  2001;95:503–506. doi:10.3171/jns.2001.95.3.0503. **Abs**.
  - 3 paediatric PNET cases: granulomas from Gelfoam or Surgicel 2–7 months after surgery.
  - Neither early nor glioma.
- de Verdier MC et al. The 2024 Brain Tumor Segmentation (BraTS) Challenge: glioma segmentation on post-treatment
  MRI. arXiv:2405.18368. **FT**, read through the arXiv HTML by a subagent.
  - A single resection-cavity label (RC) covers recent and chronic cavities.
  - Chronic cavities, older than about 3–6 months, are CSF-like on T1 and T2/FLAIR.
  - Recent cavities "often contained air, blood, and/or proteinaceous materials" with variable signal.
  - Blood has no label of its own.
  - Thin linear enhancement along the cavity and the dura is kept out of enhancing tissue (ET).
- La Rosa A et al. Temporospatial tumor dynamic changes in glioblastoma during radiotherapy. J Neurooncol
  2025;174:493–501. doi:10.1007/s11060-025-05060-7. **Abs**.
  - In a 5-patient pilot, cavity volume shrinks significantly during chemoradiation, which starts about 32 days
    after surgery.

### 1.2 Benign enhancement and the time since surgery

- Rykkje AM et al. Timing of early postoperative MRI following primary glioblastoma surgery: a retrospective study of
  contrast enhancements in 311 patients. Diagnostics 2023;13:795. doi:10.3390/diagnostics13040795. **FT**
  (PMC9955136).
  - 281/311 scans (90.4%) at 1.5 T. Enhancement patterns are analysed in the 264 patients without measurable
    tumour, and one patient can show several.
  - Before against after 48 h:
    - no enhancement 41/183 (22.4%) against 7/81 (8.6%);
    - thin linear 93/183 (50.8%) against 56/81 (69.1%).
    - The abstract's "99/183" is a typo; Table 1 gives 93/183.
  - At 0–36 h, 37–59 h and 60 h or later:
    - no enhancement 22/60, 25/182, 1/22;
    - thin linear 40.0%, 59.3%, 77.3%;
    - diffuse 11.7%, 18.7%, 45.5%;
    - nodular 20.0%, 31.3%, 31.8%.
- Lescher S et al. Time window for postoperative reactive enhancement after resection of brain tumors: less than 72
  hours. Neurosurg Focus 2014;37:E3. doi:10.3171/2014.9.FOCUS14479. **Abs**.
  - 46 high-grade gliomas at 3 T.
  - Reactive enhancement in 15 (32.6%), in 13 (28.3%) within 72 h; the earliest at 22 h 57 min.
  - Pattern: marginal or leptomeningeal/dural.
- Majós C et al. Early post-operative magnetic resonance imaging in glioblastoma: correlation among radiological
  findings and overall survival in 60 patients. Eur Radiol 2016;26:1048–1055. doi:10.1007/s00330-015-3914-x.
  **Abs**.
  - Some enhancement in 59/60.
  - Thin linear is the favourable pattern; thick linear and nodular go with shorter survival.
- Ekinci G et al. Early-postoperative magnetic resonance imaging in glial tumors: prediction of tumor regrowth and
  recurrence. Eur J Radiol 2003;45:99–107. doi:10.1016/s0720-048x(02)00027-x. **Abs**.
  - 50 gliomas scanned within 24 h; resection-bed enhancement in 45/50.
  - All residual tumours were thick linear (4) or thick linear-nodular (16); none was thin linear.
- Albert FK et al. Early postoperative magnetic resonance imaging after resection of malignant glioma. Neurosurgery
  1994;34:45–61. doi:10.1097/00006123-199401000-00008. **Abs**.
  - Imaging on days 1–3 "avoided surgically induced contrast enhancement", in the era of older, lower-field scanners.
  - Residual enhancement was the strongest prognostic factor.
- Farace P et al. Early MRI changes in glioblastoma in the period between surgery and adjuvant therapy. J
  Neurooncol 2013;111:177–185. doi:10.1007/s11060-012-0997-y. **Abs**.
  - New enhancement by about day 30 in 17/37; in 3 of them it was confined to early DWI lesions.
- Masuda Y et al. Evaluation of the extent of resection and detection of ischemic lesions with intraoperative MRI in
  glioma surgery. J Neurosurg 2019;131:209–216. doi:10.3171/2018.3.JNS172516. **Abs**.
  - Early MRI within 24 h.
  - New enhancement outside the original tumour in 12/22 enhancing tumours, and new T2 high signal in 7/8
    non-enhancing ones.
  - DWI ischaemia in 16/30.

### 1.3 Ischaemia on DWI

- Ulmer S et al. Clinical and radiographic features of peritumoral infarction following resection of glioblastoma.
  Neurology 2006;67:1668–1670. doi:10.1212/01.wnl.0000242894.21705.3c. **Abs**.
  - Restricted diffusion next to high-grade cavities in 70%.
  - 91% of these became cystic, and 43% enhanced later.
- Smith JS et al. Serial diffusion-weighted magnetic resonance imaging in cases of glioma. J Neurosurg
  2005;103:428–438. doi:10.3171/jns.2005.103.3.0428. **Abs**.
  - 28/44 (64%).
  - Enhancement appeared between day 15 and day 198; encephalomalacia followed in 26/28.
  - Clinical and surgical parameters did not predict it.
- Gempt J et al. Postoperative ischemic changes following resection of newly diagnosed and recurrent gliomas and
  their clinical relevance. J Neurosurg 2013;118:801–808. doi:10.3171/2012.12.JNS12125. **Abs** ("Gempt 2013a").
  - 26/84 (31%) newly diagnosed against 20/25 (80%) recurrent.
- Gempt J et al. Postoperative ischemic changes after glioma resection identified by diffusion-weighted magnetic
  resonance imaging and their association with intraoperative motor evoked potentials. J Neurosurg
  2013;119:829–836. doi:10.3171/2013.5.JNS121981. **Abs** ("Gempt 2013b").
  - 32/70 (46%) in motor-eloquent gliomas.
- Strand PS et al. Brain infarctions after glioma surgery: prevalence, radiological characteristics and risk
  factors. Acta Neurochir 2021;163:3097–3108. doi:10.1007/s00701-021-04914-z. **Abs**.
  - 238/539 (44%): rim-shaped 36%, sector-shaped 43%.
  - More often with temporal tumours.
  - Not more common after recurrent surgery, which conflicts with Gempt 2013a.
  - The rate varies by surgeon (32–72%).

### 1.4 Imaging protocols and guidelines

- Ellingson BM et al. Consensus recommendations for a standardized Brain Tumor Imaging Protocol in clinical trials.
  Neuro Oncol 2015;17:1188–1198. doi:10.1093/neuonc/nov095. **Abs**.
  - Minimum sequences: parameter-matched pre- and post-contrast 3D T1, T2, FLAIR and DWI.
- RANO 2010 and RANO 2.0 on post-op timing, and EANO 2021's 24–48 h window: **NV**. The report quotes them second-hand.

### 1.5 Datasets used by this project

- Cepeda S et al. The Río Hortega University Hospital Glioblastoma dataset (RHUH-GBM). Data Brief 2023;50:109617.
  doi:10.1016/j.dib.2023.109617. **FT** (PMC10551826).
  - 40 glioblastoma patients: GTR 27, NTR 13.
  - Early post-op scan within 72 h.
  - Every scanner is 1.5 T: GE Signa HDxt for 107/120 studies, Philips for the rest.
  - 5-ALA in 40/40, fluorescein in 7.
  - Labels from DeepMedic, corrected by two neurosurgeons: 1 necrosis, 2 peritumoural signal alteration (oedema and
    non-enhancing tumour), 3 enhancing tumour.
  - Mean post-op enhancing volume 0.23 cm³.
  - No rule for benign rim enhancement is stated.
  - In this project's case manifest, 20 of the 24 accepted training patients have a post-op day estimate of 1 (2 at
    day 0, 1 at day 2, 1 at day 3).
- Suter Y et al. The LUMIERE dataset: longitudinal glioblastoma MRI with expert RANO evaluation. Sci Data 2022.
  doi:10.1038/s41597-022-01881-7. Verified on 2026-11-16; see EXP-2026-005.
  - Timing is given in weeks from the pre-op scan, so the same-week pairs are at most 6 days after the pre-op scan.
  - The centre's protocol defines early post-op MRI as under 72 h (Schucht 2013, doi:10.1371/journal.pone.0079846),
    but per-scan days are not released.
  - The report's "1–3 weeks" comes from a secondary study of LUMIERE's post-op scans in general. It does not
    describe the same-week pairs used here.

### 1.6 Corrections to the deep-research report

- "Ølstad 2024" is Johnstad et al. 2024.
- Masuda 2019's early MRI was within 24 h, not at 48–72 h.
- RHUH's released labels use 3 for enhancing tumour, not 4.
- "Ischaemia occurs more often after recurrent surgery" rests on Gempt 2013a alone; Strand 2021 found no such
  difference.
- Not verified by Claude Code:
  - Bette 2016, Smets 2013 and Learned 2014;
  - Lescher's 13/46 against 15/46 discrepancy;
  - Oser 1997 and Miskin 2020;
  - the MU-Glioma-Post labels, Helland/Bouget 2023, and the Neuro-Oncology Practice meta-analysis (pooled 48%);
  - the subtraction image BraTS gave its annotators, Leao 2020, and UCSD-PTGBM.

## 2. Synthesis and segmentation methods

- Kang B et al. Pre-to-post operative MRI generation with retrieval-based visual in-context learning. MICCAI 2025.
  doi:10.1007/978-3-032-04927-8_51. **FT** (open-access PDF).
  - Data: LUMIERE, 71 pre/post pairs in MNI space, as T1ce 2D slices (2,530 train, 205 test).
  - Method:
    - it retrieves the training patient with the highest pre-op mask IoU;
    - it pastes that patient's real post-op region into the query's tumour box as a prompt (MedSAM encoder);
    - it adds a visual in-context grid.
  - It generates the image only, with no mask, and is evaluated by SSIM, MS-SSIM, PSNR and LPIPS, with no
    segmentation.
  - Haemorrhage is named but not modelled.
  - Status here: conceptual reference.
- Pérez-García F et al. A self-supervised learning strategy for postoperative brain cavity segmentation simulating
  resections. IJCARS 2021. arXiv:2105.11239. **FT**, read through ar5iv by a subagent.
  - The simulated cavity is filled with per-voxel Gaussian noise using the mean and SD of the patient's own
    ventricles, blended in with a smoothed alpha map.
  - It models CSF only. The patients had epilepsy surgery, and the scan timing is not stated.
- Billot B et al. SynthSeg: segmentation of brain MRI scans of any contrast and resolution without retraining. Med
  Image Anal 2023;86:102789. doi:10.1016/j.media.2023.102789. **Abs**.
  - Trained on images sampled from label maps with fully randomised contrast and resolution (domain randomisation).
