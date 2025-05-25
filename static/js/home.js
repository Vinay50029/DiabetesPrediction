emailjs.init("23sCj987ynMGoFeJp"); // Replace with your actual public key
  
function setActionType(actionType) {
  const form = document.getElementById("userForm");
  const name = form.elements["Name"].value;
  const email = form.elements["email"].value;

  document.getElementById("action_type").value = actionType;
  console.log("Action Type Set To:", actionType);

  if (form.checkValidity()) {
    // Send email before submitting
    const params = {
      name: name,
      email: email,
    };

    emailjs.send("service_1sq52c4", "template_8g4a2gr", params)
      .then(function(response) {
        console.log("Email sent successfully!", response);
        form.submit();  // Submit after email is sent
      }, function(error) {
        console.error("Email sending failed:", error);
        form.submit();  // Still submit even if email fails
      });
  } else {
    form.reportValidity();
  }
}