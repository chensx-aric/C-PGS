# ACA dataset

The ACA dataset combines **third-party source images obtained through the CASIA access process** with **pixel-level annotations created by the C-PGS authors**. Dataset images and annotations are intentionally not hosted in this source-only GitHub repository.

Source-image portal: <http://vision.ia.ac.cn/data/>  
Data-access contact: `wgao@nlpr.ia.ac.cn`

The portal is operated by the Institute of Automation, Chinese Academy of Sciences. The address above is solely the access contact for the CASIA source images. Reviewers and users should obtain those images through the data owner. The C-PGS repository does not claim ownership of them.

## Access and redistribution

Obtain source images through the portal/contact above and follow the data owner's terms. The author-created annotations are not publicly released; the validation materials required for GRSI evaluation are handled separately through the confidential review channel.

Expected layout:

```text
data/ACA/
├── Fayu_temple_of_mount_Puto_voc/
│   ├── JPEGImages/
│   ├── SegmentationClassNpy/
│   └── InstanceClassNpy/
├── ... five other architectural-complex directories ...
```

The split manifests in `splits/aca` use paths relative to `data/ACA`. The retained paper counts are:

| Split | Images |
|---|---:|
| training pool | 1,308 |
| labeled 1/16 | 82 |
| labeled 1/8 | 164 |
| labeled 1/4 | 327 |
| labeled 1/2 | 654 |
| validation | 145 |

The no-argument model evaluator expects this tree at `data/ACA`; see the repository README for the exact command.
