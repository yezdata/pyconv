Robustni deduplikace pro ostreni overlap duplikaci ci sldiing window context opakovani z Transcirber modulu


dodal jsem k vystupu audio loader take bool zdali se jedna o useknuty text podle max segment len - cili je pritomen overlap ci ne. To se pak v transciber pouziva k urceni toho zdali nechat tecky ktere pri umelem rezu mohou byt nakonec uprostred vety, ktera whisper halucinuje a pridava navic kdyz jsou slova useknuta pomoci max segment len a ne prirozene dle znacky konec mluvy dle VAD. - Pri VAD useknuti se tedy tecky ponechavaji.

overlap patří k nucenému střihu (dlouhý segment), ale nepatří ke konci promluvy (ticho). Tím šetříš šířku pásma a výpočetní čas Whisperu v Modulu 2. -> nedelam overlap pro KOnec continuous mluvy protoze bych pak musel posilat ticho mezi END a novym START ci nejak narusoval proudeni casu umele