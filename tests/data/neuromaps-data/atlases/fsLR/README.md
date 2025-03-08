# fsLR

The midthickness, inflated, and very inflated surfaces are from the Conte69 atlas.
The sulcal depth maps were all generated from the 164k surface via:

```bash
wb_command -metric-resample ${sulc164k} ${164k_sphere} ${Xk_sphere} ADAP_BARY_AREAS ${sulc_Xk} \
           -area-metrics ${164k_vaavg} ${Xk_vaaavg}
```

Where `${Xk_sphere}` is the target sphere and `${164k_vaavg}` + `${Xk_vaavg}` are average vertex area maps.
