# from flask import session
import streamlit as st
# from genFXLeadsCrew import genFXLeadsCrew
from gen_fxleads_crew.crew import GenFxleadsCrew

class FXLeadsGenUI:

    def _generate_leads(self, industry, material, region, expected_num, offering):
        leads_details = {
            'industry': industry,
            'material': material,
            'region': region,
            'expected_num': expected_num,
            "offering": offering,  # NDF : FX Forward
        }
        return GenFxleadsCrew().crew().kickoff(inputs=leads_details)

    def leads_generation(self):
        st.container().empty()

        if st.session_state["generating"]:
            st.write("Crew start working...chain of thoughts shown below:")
            st.session_state["leads"] = self._generate_leads(
                st.session_state["industry"],
                st.session_state["material"],
                st.session_state["region"],
                st.session_state["expected_num"],
                st.session_state["offering"],
            )
 
        if st.session_state["leads"] and st.session_state["leads"] != "":
            with st.container():
                st.write("Leads generated successfully!")
                st.download_button(
                    label="Download Leads",
                    data=st.session_state["leads"],
                    file_name="leads.html",
                    mime="text/html",
                )
                st.write("Leads Preview")
                st.markdown(st.session_state["leads"])
                # st.components.v1.html(
                #     f"""
                #     <div style="position: relative>
                #         <div style="overflow-y: scroll; height: 400px;
                #         corner-radius: 10px; padding: 20px;">
                #         {st.session_state["leads"]}
                #         </div>
                #         <div class="gradient" style="position: absolute;
                #         bottom: 0px; left: 0px; width: 100%; height: 150px;
                #         pointer-events: none;
                #         background: linear-gradient(to top,
                #         rgba(255,255,255,1), rgba(255,255,255,0));">
                #         </div>
                #     </div>
                #     """,
                #     height=500,
                # )
            
            st.session_state["generating"] = False

    def sidebar(self):
        with st.sidebar:
            st.title('FX Leads Generator')

            st.write("""
                This is a tool designed to be used by the FX Trading team to generate leads of potential clients.\n
                The leads are generated based on below parameters:\n
                """
                )
            
            st.text_input("Industry", key="industry", placeholder="Enter Industry e.g. Healthcare")
            st.text_input("Material", key="material", placeholder="Enter Material e.g. silver")
            st.text_input("Region", key="region", placeholder="Enter Region e.g. China")
            st.number_input("Expected Number of Leads", key="expected_num", min_value=0, max_value=10, placeholder="Number of Leads e.g. 0-10")
            st.text_input("Offering", key="offering", placeholder="Enter Offering e.g. FX Forward")

            if st.button("Generate Leads"):
                st.session_state["generating"] = True

    def render(self):
        st.set_page_config(page_title="FX Leads Generation", page_icon="🧊", layout='wide')

        # initialize session state variables
        if "industry" not in st.session_state:
            st.session_state["industry"] = "Mining"
        if "material" not in st.session_state:
            st.session_state["material"] = "silver"
        if "region" not in st.session_state:
            st.session_state["region"] = "China"
        if "expected_num" not in st.session_state:
            st.session_state["expected_num"] = 3
        if "offering" not in st.session_state:
            st.session_state["offering"] = "FX options"

        if "leads" not in st.session_state:
            st.session_state["leads"] = ""

        if "generating" not in st.session_state:
            st.session_state["generating"] = False

        # initialize the sidebar
        self.sidebar()

        # initialize the main content
        self.leads_generation()



if __name__ == "__main__":
    FXLeadsGenUI().render()

