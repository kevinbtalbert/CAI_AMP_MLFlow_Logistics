# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
# ###########################################################################

# CML Application script — launches the Streamlit classifier demo.
# The $CDSW_APP_PORT environment variable is set automatically by CML.
!streamlit run app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1 --server.enableCORS=false --server.enableXsrfProtection=false
