"""
Interface with cpymad to run a sequence with MAD-X
"""
import logging
from typing import Any, Optional

import cpymad.madx

from georges_core import Kinematics as _Kinematics
from georges_core.sequences import Sequence as _Sequence

APERTURE_CONVENTION = {
    "RECTANGULAR": "RECTANGLE",
    "CIRCULAR": "CIRCLE",
}


class MadX(cpymad.madx.Madx):  # type: ignore[misc]
    """
    TODO

    Args:
        cpymad (_type_): cpymad instance
    """

    def __init__(
        self, sequence: Optional[_Sequence] = None, kinematics: Optional[_Kinematics] = None, *args: Any, **kwargs: Any
    ):
        """
        TODO

        Args:
            sequence (Optional[_Sequence], optional): sequence to run. Defaults to None.
            kinematics (Optional[_Kinematics], optional): kinematics of the sequence. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.send_sequence(sequence, kinematics)

    def send_sequence(self, sequence: Optional[_Sequence] = None, kinematics: Optional[_Kinematics] = None) -> None:
        if sequence is None:
            return
        if kinematics is None:
            kinematics = sequence.kinematics
        df = sequence.to_df(strip_units=True)
        # Change apertype and aperture for MAD-X
        try:
            df["APERTYPE"] = df["APERTYPE"].apply(lambda e: APERTURE_CONVENTION[e])
            df["APERTURE"] = df["APERTURE"].apply(lambda e: set(e))
        except KeyError:
            pass

        if kinematics is not None:
            self.input(
                f"""
        BEAM, PARTICLE={kinematics.particule.__name__}, ENERGY={kinematics.etot.m_as('GeV')};
    """.strip(),
            )
        sequence_name = sequence.name.lower()
        if sequence_name == "sequence":
            logging.warning("Name of the sequence is 'sequence' which is a protected name in MAD-X. Convert to 'seq'")
            sequence_name = "seq"
        self.input(f"{sequence_name or 'SEQ'}: SEQUENCE, L={df.iloc[-1]['AT_EXIT']}, REFER=ENTRY;")
        for name, element in df.iterrows() or []:
            parameters = dict(
                element[
                    list(
                        set(list(element.index.values)).intersection(
                            set(
                                list(
                                    map(
                                        lambda _: _.upper(),  # type: ignore[no-any-return]
                                        self._libmadx.get_defined_command(element["CLASS"].lower())["data"].keys(),
                                    ),
                                ),
                            ),
                        ),
                    )
                ],
            )
            self.input(
                (
                    f"{name}: {element['CLASS'].lower()}, AT={element['AT_ENTRY']}, "
                    + ", ".join([f"{k}={str(v).strip('([])')}" for k, v in parameters.items()])
                    + ";"
                ).strip(),
            )
        self.input("ENDSEQUENCE;")
        self.input(f"USE, SEQUENCE={sequence_name or 'SEQ'};")
