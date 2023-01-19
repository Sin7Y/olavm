use plonky2::plonky2::util::serialization;
use super::proof::*;

impl Buffer {
    fn write_stark_opening_set<F: RichField + Extendable<D>, const D: usize>(&mut self, sos: &StarkOpeningSet<F, D>) -> Result<()> {
        self.write_field_ext_vec::<F, D>(&sos.local_values);
        self.write_field_ext_vec::<F, D>(&sos.next_values);
        self.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs);
        self.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs_next);
        self.write_field_vec(&sos.ctl_zs_last);
        self.write_field_ext_vec::<F, D>(&sos.quotient_polys);
    }

    fn read_stark_opening_set<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        common_data: &CommonCircuitData<F, C, D>,
    ) -> Result<OpeningSet<F, D>> {
        
    }
}